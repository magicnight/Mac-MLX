const apexOrigin = "https://macmlx.app";

export function redirectDestination(requestURL) {
  const source = new URL(requestURL);
  const destination = new URL(apexOrigin);
  destination.pathname = source.pathname;
  destination.search = source.search;
  return destination.toString();
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
