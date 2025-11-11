import type { Metadata } from "next";
import "./globals.css";
import { TransactionProvider } from "../context/TransactionContext";
import { ClerkProvider } from "@clerk/nextjs";
export const metadata: Metadata = {
  title: "Transaction Analyzer",
  description: "Analyze your Google Pay transactions",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <ClerkProvider>
          <TransactionProvider>{children}</TransactionProvider>
        </ClerkProvider>
      </body>
    </html>
  );
}
