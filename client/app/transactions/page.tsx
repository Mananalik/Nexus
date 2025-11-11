"use client";

import React, {
  Dispatch,
  SetStateAction,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useRouter } from "next/navigation";
import { useTransactions } from "../../context/TransactionContext";
import { DateRange } from "react-day-picker";
import { DatePickerWithRange } from "../components/DatePicker";

type Transaction = {
  type: string;
  amount: number;
  receiver: string;
  date: string;
  category: string;
  date_original?: string;
};

type SortKey = "date" | "amount" | "description" | "category";
type SortConfig = {
  key: SortKey;
  direction: "asc" | "desc";
};

type SortableHeaderProps = {
  label: string;
  sortKey: SortKey;
  sortConfig: SortConfig;
  setSortConfig: Dispatch<SetStateAction<SortConfig>>;
};

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
  const { transactions, parseStats } = useTransactions();
  const router = useRouter();

  const [filter, setFilter] = useState("");
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: "date",
    direction: "desc",
  });
  const [activeFilter, setActiveFilter] = useState("All");

  const [visibleCount, setVisibleCount] = useState(10);

  const [dateRange, setDateRange] = useState<DateRange | undefined>();

  useEffect(() => {
    if (transactions.length === 0) {
      router.push("/upload-activity");
    }
  }, [transactions, router]);

  useEffect(() => {
    setVisibleCount(10);
  }, [filter, sortConfig, activeFilter]);

  const processedData = useMemo(() => {
    const typeFiltered = transactions.filter((tx) => {
      if (activeFilter === "All") return true;
      if (activeFilter === "Sent")
        return tx.type === "Paid" || tx.type === "Sent";
      if (activeFilter === "Received") return tx.type === "Received";
      return true;
    });

    const filteredBySearch = typeFiltered.filter((tx: Transaction) => {
      const searchText = `${tx.receiver || ""} ${
        tx.category || ""
      }`.toLowerCase();
      return searchText.includes(filter.toLowerCase());
    });

    const filteredByDate = filteredBySearch.filter((tx) => {
      if (!dateRange?.from) return true;

      const txDate = new Date(tx.date_original || tx.date);
      const fromDate = new Date(dateRange.from);
      const toDate = dateRange.to ? new Date(dateRange.to) : fromDate;

      return txDate >= fromDate && txDate <= toDate;
    });

    const sorted = [...filteredByDate].sort((a, b) => {
      if (sortConfig.key === "date") {
        return sortConfig.direction === "asc"
          ? new Date(a.date).getTime() - new Date(b.date).getTime()
          : new Date(b.date).getTime() - new Date(a.date).getTime();
      }
      if (sortConfig.key === "amount") {
        return sortConfig.direction === "asc"
          ? a.amount - b.amount
          : b.amount - a.amount;
      }
      if (sortConfig.key === "description") {
        return sortConfig.direction === "asc"
          ? (a.receiver || "").localeCompare(b.receiver || "")
          : (b.receiver || "").localeCompare(a.receiver || "");
      }
      if (sortConfig.key === "category") {
        return sortConfig.direction === "asc"
          ? (a.category || "").localeCompare(b.category || "")
          : (b.category || "").localeCompare(a.category || "");
      }
      return 0;
    });

    const totalSent = filteredByDate
      .filter((tx) => tx.type === "Paid" || tx.type === "Sent")
      .reduce((sum, tx) => sum + tx.amount, 0);

    const totalReceived = filteredByDate
      .filter((tx) => tx.type === "Received")
      .reduce((sum, tx) => sum + tx.amount, 0);

    return {
      filteredAndSorted: sorted,
      summary: {
        totalSent,
        totalReceived,
        netFlow: totalReceived - totalSent,
      },
    };
  }, [transactions, filter, sortConfig, activeFilter, dateRange]);

  if (transactions.length === 0) {
    return (
      <div className="bg-[#222831] text-[#EEEEEE] min-h-screen font-sans flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl mb-4">Loading transactions...</p>
          <p className="text-gray-400">Redirecting to upload page...</p>
        </div>
      </div>
    );
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

  // --- ADD THIS VARIABLE ---
  // Get the slice of transactions to render
  const visibleTransactions = processedData.filteredAndSorted.slice(
    0,
    visibleCount
  );
  // --- END ---

  return (
    <div className="bg-[#222831] text-[#EEEEEE] min-h-screen font-sans">
      <main className="container mx-auto px-4 py-8 sm:py-16">
        <header className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-3">
            Transaction Analysis
          </h1>
          <p className="text-gray-400">
            {/* --- MODIFY THIS LINE --- */}
            Showing {visibleTransactions.length} of{" "}
            {processedData.filteredAndSorted.length} matching transactions (
            {transactions.length} total)
            {/* --- END --- */}
          </p>
          {parseStats && (
            <p className="text-sm text-gray-500 mt-1">
              Date parsing: {parseStats.success_rate}% success rate
            </p>
          )}
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg text-center">
            <h3 className="text-gray-400 text-sm font-bold uppercase">
              Total Sent
            </h3>
            <p className="text-3xl font-bold text-red-400 mt-2">
              ₹{processedData.summary.totalSent.toFixed(2)}
            </p>
          </div>
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg text-center">
            <h3 className="text-gray-400 text-sm font-bold uppercase">
              Total Received
            </h3>
            <p className="text-3xl font-bold text-green-400 mt-2">
              ₹{processedData.summary.totalReceived.toFixed(2)}
            </p>
          </div>
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg text-center">
            <h3 className="text-gray-400 text-sm font-bold uppercase">
              Net Flow
            </h3>
            <p
              className={`text-3xl font-bold mt-2 ${
                processedData.summary.netFlow >= 0
                  ? "text-green-400"
                  : "text-red-400"
              }`}
            >
              {processedData.summary.netFlow < 0 && "-"}₹
              {Math.abs(processedData.summary.netFlow).toFixed(2)}
            </p>
          </div>
        </div>

        <div className="flex space-x-2 mb-4">
          <button
            onClick={() => setActiveFilter("All")}
            className={`px-4 py-2 text-sm font-bold rounded-lg transition-colors ${
              activeFilter === "All"
                ? "bg-[#00ADB5] text-white"
                : "bg-[#393E46] hover:bg-[#4a505a]"
            }`}
          >
            All ({transactions.length})
          </button>
          <button
            onClick={() => setActiveFilter("Sent")}
            className={`px-4 py-2 text-sm font-bold rounded-lg transition-colors ${
              activeFilter === "Sent"
                ? "bg-red-500/80 text-white"
                : "bg-[#393E46] hover:bg-[#4a505a]"
            }`}
          >
            Sent (
            {
              transactions.filter((t) => t.type === "Paid" || t.type === "Sent")
                .length
            }
            )
          </button>
          <button
            onClick={() => setActiveFilter("Received")}
            className={`px-4 py-2 text-sm font-bold rounded-lg transition-colors ${
              activeFilter === "Received"
                ? "bg-green-500/80 text-white"
                : "bg-[#393E46] hover:bg-[#4a505a]"
            }`}
          >
            Received ({transactions.filter((t) => t.type === "Received").length}
            )
          </button>
        </div>

        <div className="mb-6 flex items-center justify-between gap-2.5">
          <input
            type="text"
            placeholder="🔍 Search by description or category..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="flex-1 p-3 bg-[#393E46] rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-[#00ADB5] text-white placeholder-gray-400"
          />

          <DatePickerWithRange
            date={dateRange}
            onDateChange={setDateRange}
            className="shrink-0"
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
                  label="Category"
                  sortKey="category"
                  sortConfig={sortConfig}
                  setSortConfig={setSortConfig}
                />
                <SortableHeader
                  label="Amount"
                  sortKey="amount"
                  sortConfig={sortConfig}
                  setSortConfig={setSortConfig}
                />
              </tr>
            </thead>
            <tbody>
              {/* --- MODIFY THIS LINE --- */}
              {visibleTransactions.map((tx, index) => (
                // --- END ---
                <tr
                  key={index}
                  className="border-b border-gray-700 last:border-b-0 hover:bg-[#4a505a] transition-colors"
                >
                  <td className="p-4 whitespace-nowrap">
                    <span className="font-semibold">{tx.date}</span>
                  </td>
                  <td className="p-4 font-medium">{tx.receiver || "-"}</td>
                  <td className="p-4">
                    <span
                      className={`px-2 py-1 text-xs font-bold rounded-full ${getTypeClass(
                        tx.type
                      )}`}
                    >
                      {tx.type}
                    </span>
                  </td>
                  <td className="p-4">{tx.category || "Miscellaneous"}</td>
                  <td className="p-4 font-mono text-right whitespace-nowrap">
                    ₹{tx.amount.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {processedData.filteredAndSorted.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <p className="text-xl">No transactions match your filter</p>
            <p className="mt-2">Try adjusting your search or filter</p>
          </div>
        )}

        {/* --- MODIFIED THIS SECTION --- */}
        {/* Pagination Buttons */}
        <div className="text-center mt-8 flex justify-center space-x-4">
          {/* "Show Less" Button */}
          {visibleCount > 10 && (
            <button
              onClick={() => setVisibleCount(10)}
              className="bg-gray-500 text-white font-bold py-2 px-6 rounded-lg hover:bg-gray-600 transition-colors"
            >
              Show Less
            </button>
          )}
          {/* "Show More" Button */}
          {processedData.filteredAndSorted.length > visibleCount && (
            <button
              onClick={() => setVisibleCount((prevCount) => prevCount + 20)}
              className="bg-[#00ADB5] text-white font-bold py-2 px-6 rounded-lg hover:bg-[#008a90] transition-colors"
            >
              Show 20 More
            </button>
          )}
        </div>
        {/* --- END MODIFICATION --- */}

        <div className="text-center mt-12 flex justify-center space-x-4">
          <button
            onClick={() => router.push("/upload-activity")}
            className="bg-gray-500 text-white font-bold py-2 px-6 rounded-lg hover:bg-gray-600 transition-colors"
          >
            Upload Another File
          </button>
          <button
            onClick={() => router.push("/dashboard")}
            className="bg-[#00ADB5] text-white font-bold py-2 px-6 rounded-lg hover:bg-[#008a90] transition-colors"
          >
            View Dashboard
          </button>
        </div>
      </main>
    </div>
  );
}
