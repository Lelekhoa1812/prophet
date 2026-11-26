// Fork, Rewind and Copy at a user message (docs/prophet/fork_rewind_copy.md).

// The text a rewind or a fork offers back in the message box. It travels with the navigation, so the notebook that opens
// next can show it as a draft. It is read without being taken, so React's double effects in development cannot lose it,
// and cleared once the person sends or the draft is replaced.
const offered = new Map<string, string>();
export const offerDraft = (notebookId: string, text: string) => void offered.set(notebookId, text);
export const peekDraft = (notebookId: string): string | undefined => offered.get(notebookId);
export const clearDraft = (notebookId: string) => void offered.delete(notebookId);

// Puts text on the clipboard. The async API needs a secure page; the fallback keeps it working on a plain http address.
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      document.body.removeChild(area);
    }
  }
}
