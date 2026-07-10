import { pages } from "./content/pages.mjs";

export const homeRoutes = Object.freeze({ en: "/", "zh-Hans": "/zh/" });

const home = Object.freeze({ id: "home", kind: "home", paths: homeRoutes });

export const routes = Object.freeze([home, ...pages]);
