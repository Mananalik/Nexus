// app/transactions/page.tsx
"use client";

import { useTransactions } from "../../context/TransactionContext";
import { useRouter } from "next/navigation";
import { Dispatch, SetStateAction, useEffect, useMemo, useState } from "react";

// --- START: TYPE DEFINITIONS ---
// Defined a robust Transaction type, allowing for undefined properties
type Transaction = {
  type: string;
  amount: number | null;
  receiver?: string | null; // Made optional to allow undefined
  account?: string | null; // Made optional to allow undefined
  date: string;
  time: string;
};

// Defined types for sorting state management
type SortKey = "date" | "amount" | "description";
type SortConfig = {
  key: SortKey;
  direction: "asc" | "desc";
};

// Defined props for the SortableHeader component
type SortableHeaderProps = {
  label: string;
  sortKey: SortKey;
  sortConfig: SortConfig;
  setSortConfig: Dispatch<SetStateAction<SortConfig>>;
};
// --- END: TYPE DEFINITIONS ---

// Typed the props for the helper component
const SortableHeader: React.FC<SortableHeaderProps> = ({
  label,
  sortKey,
  sortConfig,
  setSortConfig,
}) => {
  const isSorted = sortConfig.key === sortKey;
  const direction = isSorted ? sortConfig.direction : "none";

  const handleClick = () => {
    let newDirection: "asc" | "desc" = "desc";
    if (isSorted && sortConfig.direction === "desc") {
      newDirection = "asc";
    }
    setSortConfig({ key: sortKey, direction: newDirection });
  };

  const icon = direction === "asc" ? "▲" : direction === "desc" ? "▼" : "↕";

  return (
    <th
      className="p-4 font-semibold cursor-pointer select-none"
      onClick={handleClick}
    >
      {label} <span className="text-gray-400 text-xs">{icon}</span>
    </th>
  );
};

export default function TransactionsPage() {
  const { transactions } = useTransactions();
  const router = useRouter();

  const [filter, setFilter] = useState("");
  // Explicitly typed the useState for sortConfig
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: "date",
    direction: "desc",
  });

  useEffect(() => {
    if (transactions.length === 0) {
      router.push("/");
    }
  }, [transactions, router]);

  const processedData = useMemo(() => {
    // Ensure transactions is an array of our defined Transaction type
    const txs: Transaction[] = transactions || [];

    const parseDate = (tx: Transaction) => {
      return new Date(`${tx.date} ${tx.time}`);
    };

    const filtered = txs.filter((tx: Transaction) => {
      const description = `${tx.receiver || ""} ${
        tx.account || ""
      }`.toLowerCase();
      return description.includes(filter.toLowerCase());
    });

    const sorted = [...filtered].sort((a, b) => {
      if (sortConfig.key === "date") {
        const dateA = parseDate(a);
        const dateB = parseDate(b);
        // FIX: Use .getTime() for safe date comparison
        return sortConfig.direction === "asc"
          ? dateA.getTime() - dateB.getTime()
          : dateB.getTime() - dateA.getTime();
      }
      if (sortConfig.key === "amount") {
        const amountA = a.amount || 0;
        const amountB = b.amount || 0;
        return sortConfig.direction === "asc"
          ? amountA - amountB
          : amountB - amountA;
      }
      if (sortConfig.key === "description") {
        const descA = a.receiver || "";
        const descB = b.receiver || "";
        return sortConfig.direction === "asc"
          ? descA.localeCompare(descB)
          : descB.localeCompare(descA);
      }
      return 0;
    });

    const totalSent = txs
      .filter((tx) => (tx.type === "Paid" || tx.type === "Sent") && tx.amount)
      .reduce((sum, tx) => sum + tx.amount!, 0);

    const totalReceived = txs
      .filter((tx) => tx.type === "Received" && tx.amount)
      .reduce((sum, tx) => sum + tx.amount!, 0);

    const netFlow = totalReceived - totalSent;

    return {
      filteredAndSorted: sorted,
      summary: { totalSent, totalReceived, netFlow },
    };
  }, [transactions, filter, sortConfig]);

  if (transactions.length === 0) {
    return null;
  }

  const getTypeClass = (type: string) => {
    switch (type) {
      case "Paid":
      case "Sent":
        return "bg-red-500/20 text-red-300";
      case "Received":
        return "bg-green-500/20 text-green-300";
      default:
        return "bg-gray-500/20 text-gray-300";
    }
  };

  return (
    <div className="bg-[#222831] text-[#EEEEEE] min-h-screen font-sans">
      <main className="container mx-auto px-4 py-8 sm:py-16">
        <header className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-white">
            Transaction Analysis
          </h1>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg text-center">
            <h3 className="text-gray-400 text-sm font-bold uppercase">
              Total Sent
            </h3>
            <p className="text-3xl font-bold text-red-400 mt-2">
              ₹{processedData.summary.totalSent?.toFixed(2)}
            </p>
          </div>
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg text-center">
            <h3 className="text-gray-400 text-sm font-bold uppercase">
              Total Received
            </h3>
            <p className="text-3xl font-bold text-green-400 mt-2">
              ₹{processedData.summary.totalReceived?.toFixed(2)}
            </p>
          </div>
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg text-center">
            <h3 className="text-gray-400 text-sm font-bold uppercase">
              Net Flow
            </h3>
            {/* FIX: Provide a fallback of 0 to prevent 'undefined' errors */}
            <p
              className={`text-3xl font-bold mt-2 ${
                (processedData.summary.netFlow || 0) >= 0
                  ? "text-green-400"
                  : "text-red-400"
              }`}
            >
              {(processedData.summary.netFlow || 0) < 0 && "-"}₹
              {Math.abs(processedData.summary.netFlow || 0).toFixed(2)}
            </p>
          </div>
        </div>

        <div className="mb-6">
          <input
            type="text"
            placeholder="🔍 Search by description..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full p-3 bg-[#393E46] rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-[#00ADB5]"
          />
        </div>

        <div className="bg-[#393E46] rounded-xl shadow-lg overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-[#222831]">
              <tr>
                <SortableHeader
                  label="Date"
                  sortKey="date"
                  sortConfig={sortConfig}
                  setSortConfig={setSortConfig}
                />
                <SortableHeader
                  label="Description"
                  sortKey="description"
                  sortConfig={sortConfig}
                  setSortConfig={setSortConfig}
                />
                <th className="p-4 font-semibold">Type</th>
                <SortableHeader
                  label="Amount"
                  sortKey="amount"
                  sortConfig={sortConfig}
                  setSortConfig={setSortConfig}
                />
              </tr>
            </thead>
            <tbody>
              {processedData.filteredAndSorted.map((tx, index) => (
                <tr
                  key={index}
                  className="border-b border-gray-700 last:border-b-0 hover:bg-[#4a505a] transition-colors"
                >
                  <td className="p-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <span className="font-semibold">{tx.date}</span>
                      <span className="text-xs text-gray-400">{tx.time}</span>
                    </div>
                  </td>
                  <td className="p-4 font-medium">
                    {tx.receiver ||
                      (tx.account
                        ? `Account ending in ${tx.account.slice(-4)}`
                        : "-")}
                  </td>
                  <td className="p-4">
                    <span
                      className={`px-2 py-1 text-xs font-bold rounded-full ${getTypeClass(
                        tx.type
                      )}`}
                    >
                      {tx.type}
                    </span>
                  </td>
                  <td className="p-4 font-mono text-right whitespace-nowrap">
                    {tx.amount !== null ? `₹${tx.amount.toFixed(2)}` : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="text-center mt-8">
          <button
            onClick={() => router.push("/")}
            className="bg-[#00ADB5] text-white font-bold py-2 px-6 rounded-lg hover:bg-[#008a90] transition-colors"
          >
            Upload Another File
          </button>
        </div>
      </main>
    </div>
  );
}
