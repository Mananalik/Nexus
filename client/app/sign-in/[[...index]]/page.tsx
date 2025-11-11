"use client";
import { SignIn } from "@clerk/nextjs";
import Link from "next/link";

export default function SignInPage() {
  return (
    <div
      className="flex items-center justify-center min-h-screen"
      style={{
        backgroundColor: "#222831",
        color: "#EEEEEE",
      }}
    >
      <div
        className="flex flex-col items-center justify-center p-8 rounded-2xl shadow-2xl"
        style={{
          backgroundColor: "#222831",
          boxShadow: "0 8px 25px rgba(34, 40, 49, 0.15)",
        }}
      >
        <h1
          className="text-3xl font-bold mb-6 tracking-wide"
          style={{
            color: "#00ADB5",
          }}
        >
          Welcome Back
        </h1>

        {/* Clerk Sign-In component */}
        <SignIn
          routing="path"
          path="/sign-in"
          appearance={{
            elements: {
              formButtonPrimary: {
                backgroundColor: "#00ADB5",
                color: "#FFFFFF",
                fontWeight: "600",
                "&:hover": {
                  backgroundColor: "#00a0a8",
                },
              },
              socialButtonsBlockButton: {
                backgroundColor: "#FFFFFF",
                border: "1px solid #393E46",
                color: "#222831",
                fontWeight: "500",
                "&:hover": {
                  backgroundColor: "#F5F5F5",
                },
              },
              socialButtonsProviderIcon__google: {
                filter: "none",
              },
              card: {
                backgroundColor: "transparent",
                boxShadow: "none",
              },
              formFieldInput: {
                backgroundColor: "#F7F7F7",
                color: "#222831",
                borderRadius: "8px",
                border: "1px solid #CCC",
              },
              formFieldLabel: {
                color: "#EEEEEE",
                fontWeight: "500",
              },
              footer: {
                color: "#393E46",
              },
            },
            variables: {
              colorPrimary: "#00ADB5",
              colorText: "#EEEEEE",
              colorBackground: "#222831",
              fontSize: "16px",
              borderRadius: "10px",
            },
          }}
        />

        {/* Sign-Up prompt */}
        <p className="mt-6 text-sm text-center text-gray-300">
          Don’t have an account?{" "}
          <Link
            href="/sign-up"
            className="font-semibold hover:underline"
            style={{
              color: "#00ADB5",
            }}
          >
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
