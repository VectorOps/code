import type { RunnerConnection } from "./websocket";

export type FileMatch = { path: string; highlights: [number, number][] };
export type FileSearchSnapshot = {
  index_revision: number;
  results: FileMatch[];
  refreshing: boolean;
  incomplete: boolean;
  truncated: boolean;
  refresh_id: number | null;
  refresh_error: string | null;
};

export class FileSearchApi {
  constructor(private connection: RunnerConnection) {}

  search(query: string, refresh = false): Promise<FileSearchSnapshot> {
    return this.connection.request("file_search.search", {
      query,
      limit: 20,
      refresh,
    });
  }
}
