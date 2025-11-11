"use client";

import { useEffect } from "react";
import { useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const { isLoaded, isSignedIn } = useUser();
  const router = useRouter();

  useEffect(() => {
    // Wait until Clerk finishes loading
    if (!isLoaded) return;

    if (isSignedIn) {
      // user is logged in -> go to dashboard
      router.replace("/dashboard");
    } else {
      // not signed in -> go to sign-in page
      router.replace("/sign-in");
    }
  }, [isLoaded, isSignedIn, router]);

  // optional fallback UI
  return (
    <div className="flex items-center justify-center min-h-screen bg-[#222831] text-[#EEEEEE]">
      <p>Checking authentication...</p>
    </div>
  );
}
