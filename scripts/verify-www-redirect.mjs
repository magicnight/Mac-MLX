import { pathToFileURL } from "node:url";

const redirectChecks = [
  ["https://www.macmlx.app/", "https://macmlx.app/"],
  ["https://www.macmlx.app/zh/", "https://macmlx.app/zh/"],
  [
    "https://www.macmlx.app/models/choosing-a-model/?q=mlx%20swift",
    "https://macmlx.app/models/choosing-a-model/?q=mlx%20swift",
  ],
];

async function verifyRedirect(fetchImpl, source, expectedDestination) {
  const response = await fetchImpl(source, {
    redirect: "manual",
    signal: AbortSignal.timeout(10_000),
  });

  if (response.status !== 308) {
    throw new Error(`${source}: expected status 308, received ${response.status}`);
  }

  const actualDestination = response.headers.get("location");
  if (actualDestination !== expectedDestination) {
    throw new Error(
      `${source}: expected Location ${expectedDestination}, received ${actualDestination ?? "<missing>"}`,
    );
  }
}

export async function verifyWWWRedirect({ fetchImpl = globalThis.fetch } = {}) {
  const results = await Promise.allSettled(
    redirectChecks.map(([source, destination]) =>
      verifyRedirect(fetchImpl, source, destination),
    ),
  );
  const errors = results
    .filter((result) => result.status === "rejected")
    .map((result) => result.reason);

  if (errors.length > 0) {
    throw new AggregateError(errors, "www redirect verification failed");
  }

  return { checked: redirectChecks.length };
}

const isDirectExecution = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isDirectExecution) {
  verifyWWWRedirect()
    .then(({ checked }) => {
      console.log(`Verified ${checked} www redirect checks.`);
    })
    .catch((error) => {
      console.error(error);
      process.exitCode = 1;
    });
}
