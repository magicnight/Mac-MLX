import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  acquireBuildLock,
  backupPathFor,
  recoverInterruptedOutput,
  replaceGeneratedOutput,
} from "../lib/build-output.mjs";

async function temporaryWorkspace(t) {
  const workspace = await mkdtemp(join(tmpdir(), "macmlx-site-build-"));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  return workspace;
}

test("build lock excludes concurrent builders and releases cleanly", async (t) => {
  const workspace = await temporaryWorkspace(t);
  const release = await acquireBuildLock(workspace);

  await assert.rejects(acquireBuildLock(workspace), /Another public build is already running/);
  await release();

  const releaseAgain = await acquireBuildLock(workspace);
  await releaseAgain();
});

test("build lock recovers an owner left by a terminated process", async (t) => {
  const workspace = await temporaryWorkspace(t);
  const lockDirectory = join(workspace, ".public-build.lock");
  await mkdir(lockDirectory);
  await writeFile(join(lockDirectory, "owner.json"), JSON.stringify({ pid: 2_147_483_647 }), "utf8");

  const release = await acquireBuildLock(workspace);
  await release();
});

test("build lock recovers an implausibly old owner even when its PID was reused", async (t) => {
  const workspace = await temporaryWorkspace(t);
  const lockDirectory = join(workspace, ".public-build.lock");
  await mkdir(lockDirectory);
  await writeFile(join(lockDirectory, "owner.json"), JSON.stringify({ pid: process.pid, startedAt: "2026-07-10T00:00:00.000Z" }), "utf8");

  const release = await acquireBuildLock(workspace, {
    now: () => new Date("2026-07-10T00:02:00.000Z"),
    maxLockAgeMs: 60_000,
  });
  await release();
});

test("backup paths are unique and remain inside the build workspace", async (t) => {
  const workspace = await temporaryWorkspace(t);
  const first = backupPathFor(workspace);
  const second = backupPathFor(workspace);

  assert.notEqual(first, second);
  assert.equal(first.startsWith(`${workspace}/.public-backup-`), true);
  assert.equal(second.startsWith(`${workspace}/.public-backup-`), true);
});

test("interrupted backup is recovered when public output is absent", async (t) => {
  const workspace = await temporaryWorkspace(t);
  const backup = join(workspace, ".public-backup-interrupted");
  const output = join(workspace, "public");
  await mkdir(backup);
  await writeFile(join(backup, "index.html"), "old output", "utf8");

  assert.equal(await recoverInterruptedOutput({ workspaceDirectory: workspace, outputDirectory: output }), true);
  assert.equal(await readFile(join(output, "index.html"), "utf8"), "old output");
  assert.deepEqual((await readdir(workspace)).filter((name) => name.startsWith(".public-backup-")), []);
});

test("failed replacement restores previous output", async (t) => {
  const workspace = await temporaryWorkspace(t);
  const output = join(workspace, "public");
  await mkdir(output);
  await writeFile(join(output, "index.html"), "old output", "utf8");

  await assert.rejects(
    replaceGeneratedOutput({
      workspaceDirectory: workspace,
      outputDirectory: output,
      stagedDirectory: join(workspace, "missing-stage"),
    }),
    /ENOENT/,
  );

  assert.equal(await readFile(join(output, "index.html"), "utf8"), "old output");
  assert.deepEqual((await readdir(workspace)).filter((name) => name.startsWith(".public-backup-")), []);
});
