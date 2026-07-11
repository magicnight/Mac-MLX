const apexOrigin = "https://macmlx.app";

export function redirectDestination(requestURL) {
  const source = new URL(requestURL);
  return new URL(`${source.pathname}${source.search}`, apexOrigin).toString();
}

export default {
  fetch(request) {
    return new Response(null, {
      status: 308,
      headers: {
        Location: redirectDestination(request.url),
        "Cache-Control": "public, max-age=3600",
      },
    });
  },
};
