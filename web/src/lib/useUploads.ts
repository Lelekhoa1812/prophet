import { useCallback, useState } from "react";
import { api } from "./api";
import { errorMessage } from "./format";

export type UploadNote = { id: number; name: string; message: string };

// Adds files to a notebook one after another and keeps the per-file messages. Shared by the drop zone,
// the attach button in the message box and the sources panel, so every way of adding a file behaves the same.
export function useUploads(notebookId: string, onChanged: () => void) {
  const [uploading, setUploading] = useState<{ done: number; total: number; name: string } | null>(null);
  const [notes, setNotes] = useState<UploadNote[]>([]);

  const addFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      const failed: UploadNote[] = [];
      for (let i = 0; i < files.length; i += 1) {
        const file = files[i];
        setUploading({ done: i, total: files.length, name: file.name });
        try {
          await api.uploadSource(notebookId, file);
        } catch (err) {
          failed.push({ id: Date.now() + i, name: file.name, message: errorMessage(err, "Couldn't add this file.") });
        }
      }
      setUploading(null);
      if (failed.length) setNotes((prev) => [...failed, ...prev].slice(0, 6));
      onChanged();
    },
    [notebookId, onChanged],
  );

  const note = useCallback((name: string, message: string) => {
    setNotes((prev) => [{ id: Date.now(), name, message }, ...prev].slice(0, 6));
  }, []);

  const dismiss = useCallback((id: number) => setNotes((prev) => prev.filter((n) => n.id !== id)), []);

  return { uploading, notes, addFiles, note, dismiss };
}

export type Uploads = ReturnType<typeof useUploads>;
