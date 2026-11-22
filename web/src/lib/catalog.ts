import { useEffect, useState } from "react";
import { api, type SkillDetail, type SkillInfo, type ToolInfo } from "./api";

// The playbooks and tools a "/" pick can name. Fetched once per page load and shared by the box and the sent messages,
// so a chip in the transcript can say what it is without every chip asking the server.
let skills: Promise<SkillInfo[]> | null = null;
let tools: Promise<ToolInfo[]> | null = null;
const details = new Map<string, Promise<SkillDetail>>();

export function loadSkills(): Promise<SkillInfo[]> {
  skills ??= api.skills().catch((e) => {
    skills = null;
    throw e;
  });
  return skills;
}

export function loadTools(): Promise<ToolInfo[]> {
  tools ??= api.tools().catch((e) => {
    tools = null;
    throw e;
  });
  return tools;
}

export function loadSkill(name: string): Promise<SkillDetail> {
  let p = details.get(name);
  if (!p) {
    p = api.skill(name).catch((e) => {
      details.delete(name);
      throw e;
    });
    details.set(name, p);
  }
  return p;
}

export function usePlaybooks(): SkillInfo[] {
  const [list, setList] = useState<SkillInfo[]>([]);
  useEffect(() => {
    let live = true;
    loadSkills().then((l) => live && setList(l)).catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);
  return list;
}
