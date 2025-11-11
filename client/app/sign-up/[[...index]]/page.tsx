"use client";
import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <div
      className="flex items-center justify-center min-h-screen"
      style={{
        backgroundColor: "#222831", // page background
        color: "#EEEEEE",
      }}
    >
      <div
        className="flex flex-col items-center justify-center p-8 rounded-2xl shadow-2xl"
        style={{
          backgroundColor: "#222831", // card background (keeps dark, blended)
          boxShadow: "0 8px 25px rgba(34, 40, 49, 0.15)",
        }}
      >
        <div
          style={{
            padding: 18,
            background: "rgba(57, 62, 70, 0.55)",
            borderRadius: 12,
            boxShadow: "inset 0 -6px 30px rgba(0,0,0,0.35)",
          }}
        >
          <SignUp
            routing="path"
            path="/sign-up"
            appearance={{
              elements: {
                formButtonPrimary: {
                  backgroundColor: "#00ADB5",
                  color: "#FFFFFF",
                  fontWeight: 600,
                  borderRadius: "8px",
                  "&:hover": { backgroundColor: "#00a0a8" },
                },
                socialButtonsBlockButton: {
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #393E46",
                  color: "#222831",
                  fontWeight: 500,
                  "&:hover": { backgroundColor: "#F5F5F5" },
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
                  fontWeight: 500,
                },
                footer: {
                  color: "#EEEEEE",
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
        </div>
      </div>
    </div>
  );
}
