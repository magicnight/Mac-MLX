import { randomUUID } from "node:crypto";
import { access, mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";

const lockName = ".public-build.lock";
const backupPrefix = ".public-backup-";

async function pathExists(path) {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

function processIsRunning(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code !== "ESRCH";
  }
}

async function lockIsStale(lockDirectory, { now, maxLockAgeMs }) {
  try {
    const owner = JSON.parse(await readFile(join(lockDirectory, "owner.json"), "utf8"));
    const startedAt = Date.parse(owner.startedAt);
    if (Number.isFinite(startedAt) && now().getTime() - startedAt > maxLockAgeMs) return true;
    if (!Number.isFinite(startedAt)) {
      const metadata = await stat(lockDirectory);
      if (now().getTime() - metadata.mtimeMs > maxLockAgeMs) return true;
    }
    return !processIsRunning(owner.pid);
  } catch (error) {
    if (error.code !== "ENOENT" && !(error instanceof SyntaxError)) throw error;
    const metadata = await stat(lockDirectory);
    return now().getTime() - metadata.mtimeMs > maxLockAgeMs;
  }
}

export async function acquireBuildLock(workspaceDirectory, { now = () => new Date(), maxLockAgeMs = 15 * 60_000 } = {}) {
  const lockDirectory = join(workspaceDirectory, lockName);
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      await mkdir(lockDirectory);
      try {
        await writeFile(join(lockDirectory, "owner.json"), JSON.stringify({ pid: process.pid, startedAt: now().toISOString() }), "utf8");
      } catch (error) {
        await rm(lockDirectory, { recursive: true, force: true });
        throw error;
      }

      let released = false;
      return async function releaseBuildLock() {
        if (released) return;
        released = true;
        await rm(lockDirectory, { recursive: true, force: true });
      };
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
      if (attempt === 0 && await lockIsStale(lockDirectory, { now, maxLockAgeMs })) {
        await rm(lockDirectory, { recursive: true, force: true });
        continue;
      }
      throw new Error("Another public build is already running", { cause: error });
    }
  }

  throw new Error("Unable to acquire public build lock");
}

export function backupPathFor(workspaceDirectory) {
  return join(workspaceDirectory, `${backupPrefix}${process.pid}-${randomUUID()}`);
}

async function backupDirectories(workspaceDirectory) {
  const entries = await readdir(workspaceDirectory, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isDirectory() && entry.name.startsWith(backupPrefix))
    .map((entry) => join(workspaceDirectory, entry.name));
}

export async function recoverInterruptedOutput({ workspaceDirectory, outputDirectory }) {
  const backups = await backupDirectories(workspaceDirectory);
  if (backups.length === 0) return false;

  if (await pathExists(outputDirectory)) {
    await Promise.all(backups.map((backup) => rm(backup, { recursive: true, force: true })));
    return false;
  }

  const datedBackups = await Promise.all(backups.map(async (path) => ({ path, modified: (await stat(path)).mtimeMs })));
  datedBackups.sort((left, right) => right.modified - left.modified || right.path.localeCompare(left.path));
  const [latest, ...obsolete] = datedBackups;
  await rename(latest.path, outputDirectory);
  await Promise.all(obsolete.map((backup) => rm(backup.path, { recursive: true, force: true })));
  return true;
}

export async function replaceGeneratedOutput({ workspaceDirectory, outputDirectory, stagedDirectory }) {
  // Local directory replacement uses two renames so rollback remains possible;
  // readers may briefly observe no public/ directory between them. Production
  // publication is delegated to Cloudflare's atomic deployment boundary.
  await recoverInterruptedOutput({ workspaceDirectory, outputDirectory });
  const backupDirectory = backupPathFor(workspaceDirectory);
  let hadOutput = true;
  try {
    await rename(outputDirectory, backupDirectory);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    hadOutput = false;
  }

  try {
    await rename(stagedDirectory, outputDirectory);
  } catch (error) {
    if (hadOutput) await rename(backupDirectory, outputDirectory);
    throw error;
  }

  if (hadOutput) await rm(backupDirectory, { recursive: true, force: true });
}
