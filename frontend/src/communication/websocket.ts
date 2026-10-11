import type { Params } from "../models/json";
type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
};

export class RunnerConnection {
  private socket?: WebSocket;
  private pending = new Map<string, Pending>();
  private reconnect?: ReturnType<typeof setTimeout>;
  private handshake?: ReturnType<typeof setTimeout>;
  private closed = false;
  private authenticated = false;
  private attempts = 0;
  private sequence = 0;
  onReady = () => {};
  onEvent = (_event: Params) => {};
  onStatus = (_status: string) => {};

  constructor(
    private endpoint: string,
    private token: string,
  ) {
    const url = new URL(endpoint);
    if (
      !["ws:", "wss:"].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.hash
    )
      throw new Error("Invalid WebSocket endpoint");
    if (
      url.protocol === "ws:" &&
      !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)
    )
      throw new Error("Remote connections require WSS or an SSH tunnel");
  }

  connect() {
    if (this.closed) return;
    this.onStatus("Connecting");
    const socket = new WebSocket(this.endpoint);
    this.socket = socket;
    this.handshake = setTimeout(() => socket.close(), 12000);
    socket.onopen = () =>
      socket.send(
        JSON.stringify({ kind: "authenticate", version: 2, token: this.token }),
      );
    socket.onmessage = (message) => {
      try {
        const packet = JSON.parse(String(message.data));
        if (packet.kind === "authenticated") {
          clearTimeout(this.handshake);
          this.authenticated = true;
          this.attempts = 0;
          this.sequence = 0;
          this.onStatus("Connected");
          this.onReady();
        } else if (packet.kind === "event") {
          if (packet.sequence <= this.sequence) return;
          this.sequence = packet.sequence;
          this.onEvent(packet.event);
        } else if (packet.kind === "response" || packet.kind === "error") {
          const pending = this.pending.get(packet.id);
          if (!pending) return;
          clearTimeout(pending.timeout);
          this.pending.delete(packet.id);
          if (packet.kind === "error")
            pending.reject(new Error(`${packet.code}: ${packet.message}`));
          else pending.resolve(packet.result);
        }
      } catch {
        socket.close(1002, "Invalid protocol message");
      }
    };
    socket.onclose = () => {
      clearTimeout(this.handshake);
      this.authenticated = false;
      this.rejectPending();
      if (this.closed) return;
      this.onStatus("Disconnected; reconnecting (check endpoint and token)");
      const delay =
        Math.min(1000 * 2 ** this.attempts++, 15000) + Math.random() * 500;
      this.reconnect = setTimeout(() => this.connect(), delay);
    };
    socket.onerror = () => socket.close();
  }

  request<T>(method: string, params: Params = {}): Promise<T> {
    if (!this.authenticated || this.socket?.readyState !== WebSocket.OPEN)
      return Promise.reject(new Error("Not connected"));
    const id = crypto.randomUUID();
    return new Promise<T>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(
          new Error(
            "Request timed out; outcome may be unknown. Retry unchanged input to reuse its idempotency key.",
          ),
        );
      }, 30000);
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
        timeout,
      });
      this.socket!.send(
        JSON.stringify({ kind: "request", id, method, params }),
      );
    });
  }

  private rejectPending() {
    for (const item of this.pending.values()) {
      clearTimeout(item.timeout);
      item.reject(new Error("Connection lost; command outcome may be unknown"));
    }
    this.pending.clear();
  }

  close() {
    this.closed = true;
    clearTimeout(this.reconnect);
    clearTimeout(this.handshake);
    this.socket?.close();
    this.rejectPending();
  }
}
