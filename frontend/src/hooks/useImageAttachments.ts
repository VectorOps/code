import { useEffect, useRef, useState } from "react";
import * as chat from "../models/chat-input";

type Draft = { images: chat.ImageAttachment[]; error: string };
const emptyDraft: Draft = { images: [], error: "" };

export function useImageAttachments(scope: string) {
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const latest = useRef(drafts);
  const pending = useRef(new Map<string, AbortController>());
  useEffect(
    () => () => {
      for (const controller of pending.current.values()) controller.abort();
      pending.current.clear();
    },
    [],
  );

  function update(key: string, change: (draft: Draft) => Draft) {
    const next = {
      ...latest.current,
      [key]: change(latest.current[key] ?? emptyDraft),
    };
    latest.current = next;
    setDrafts(next);
  }

  function add(files: File[]) {
    const current = latest.current[scope] ?? emptyDraft;
    if (files.length + current.images.length > chat.maxContentBlocks) {
      update(scope, (draft) => ({
        ...draft,
        error: "Attach at most four images per message.",
      }));
      return;
    }
    const images: chat.ImageAttachment[] = files.map((file) => ({
      id: crypto.randomUUID(),
      name: file.name || "Pasted image",
      state: "preparing",
    }));
    update(scope, (draft) => ({
      images: [...draft.images, ...images],
      error: "",
    }));
    files.forEach((file, index) => {
      const id = images[index].id;
      const controller = new AbortController();
      pending.current.set(id, controller);
      void chat
        .readImage(file, controller.signal)
        .then(
          (image) => {
            if (controller.signal.aborted) return;
            update(scope, (draft) => ({
              ...draft,
              images: draft.images.map((item) =>
                item.id === id
                  ? { id, name: item.name, state: "ready", image }
                  : item,
              ),
            }));
          },
          (reason: unknown) => {
            if (controller.signal.aborted) return;
            update(scope, (draft) => ({
              ...draft,
              images: draft.images.map((item) =>
                item.id === id
                  ? {
                      id,
                      name: item.name,
                      state: "failed",
                      error:
                        reason instanceof Error
                          ? reason.message
                          : "Could not read this image.",
                    }
                  : item,
              ),
            }));
          },
        )
        .finally(() => pending.current.delete(id));
    });
  }

  function remove(ids: readonly string[], key = scope) {
    for (const id of ids) {
      pending.current.get(id)?.abort();
      pending.current.delete(id);
    }
    update(key, (draft) => ({
      images: draft.images.filter((image) => !ids.includes(image.id)),
      error: "",
    }));
  }

  return { ...(drafts[scope] ?? emptyDraft), add, remove };
}
