import type { Metadata } from "next";
import "./globals.css";
import { TransactionProvider } from "../context/TransactionContext";
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
        <TransactionProvider>
          {children}
        </TransactionProvider>
      </body>
    </html>
  );
}
