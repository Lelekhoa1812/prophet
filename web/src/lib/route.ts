// Links that carry a place: /n/<id> is a notebook, /c/<id> is a quick forecast (a single chat). "/" is the start page.
// Ids are the hex ids the API hands out, so a link copied from the address bar opens the same thing in another tab.
export type Route = { kind: "n" | "c"; id: string } | null;

const PATTERN = /^\/(n|c)\/([A-Za-z0-9_-]{1,64})\/?$/;

export function parseRoute(pathname: string): Route {
  const m = PATTERN.exec(pathname);
  return m ? { kind: m[1] as "n" | "c", id: m[2] } : null;
}

export function pathFor(route: Route): string {
  return route ? `/${route.kind}/${route.id}` : "/";
}

// The address a selection should show. A notebook wins over a forecast, matching how App keeps only one selected.
export function routeFor(notebookId: string | null, missionId: string | null): Route {
  if (notebookId) return { kind: "n", id: notebookId };
  if (missionId) return { kind: "c", id: missionId };
  return null;
}
