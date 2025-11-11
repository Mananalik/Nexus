import { clerkMiddleware } from "@clerk/nextjs/server";

export default clerkMiddleware();

export const config = {
  // Keep the matcher broad so auth runs for app pages and API routes
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes
    "/(api|trpc)(.*)",
  ],
  publicRoutes: [
    "/sign-in(.*)",
    "/sign-up(.*)",
    // add other publicly accessible catch-all routes if needed
  ],
};
