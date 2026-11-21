import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import logo from "./public/logo.png";
import "@fontsource/cormorant-garamond/400.css";
import "@fontsource/cormorant-garamond/600.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "./index.css";

// The Prophet logo is the favicon too, so the browser tab matches the app.
{
  const icon = document.querySelector<HTMLLinkElement>('link[rel="icon"]') ?? document.createElement("link");
  icon.rel = "icon";
  icon.type = "image/png";
  icon.href = logo;
  document.head.appendChild(icon);
}

const root = document.getElementById("root");
if (!root) throw new Error("missing #root element");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
