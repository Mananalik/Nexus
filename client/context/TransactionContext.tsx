"use client";

import { createContext, useState, useContext, ReactNode } from "react";

interface Transaction {
  type: string;
  amount: number;
  receiver?: string | null;
  account?: string | null;
  date: string;
  time: string;
}

interface TransactionContextType {
  transactions: Transaction[];
  setTransactions: (transactions: Transaction[]) => void;
}

const TransactionContext = createContext<TransactionContextType | undefined>(
  undefined
);

export function TransactionProvider({ children }: { children: ReactNode }) {
  const [transactions, setTransactions] = useState<Transaction[]>([]);

  return (
    <TransactionContext.Provider value={{ transactions, setTransactions }}>
      {children}
    </TransactionContext.Provider>
  );
}

export function useTransactions() {
  const context = useContext(TransactionContext);
  if (context === undefined) {
    throw new Error(
      "useTransactions must be used within a TransactionProvider"
    );
  }
  return context;
}
