"use client";

import React, { createContext, useContext, useState, ReactNode } from 'react';

type Transaction = {
  type: string;
  amount: number;
  receiver: string;
  date: string;
  category: string;
  date_original?: string;
};

type ParseStats = {
  total_attempts: number;
  successful: number;
  failed: number;
  success_rate: number;
  failed_dates: Array<[string, string]>;
};

type TransactionContextType = {
  transactions: Transaction[];
  setTransactions: (transactions: Transaction[]) => void;
  parseStats: ParseStats | null;
  setParseStats: (stats: ParseStats | null) => void;
};

const TransactionContext = createContext<TransactionContextType | undefined>(undefined);

export function TransactionProvider({ children }: { children: ReactNode }) {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [parseStats, setParseStats] = useState<ParseStats | null>(null);

  return (
    <TransactionContext.Provider 
      value={{ 
        transactions, 
        setTransactions, 
        parseStats, 
        setParseStats 
      }}
    >
      {children}
    </TransactionContext.Provider>
  );
}

export function useTransactions() {
  const context = useContext(TransactionContext);
  if (context === undefined) {
    throw new Error('useTransactions must be used within a TransactionProvider');
  }
  return context;
}
