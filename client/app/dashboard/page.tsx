"use client";

import React, { useMemo, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTransactions } from "../../context/TransactionContext";
import {
  PieChart,
  Pie,
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Calendar,
  PieChart as PieChartIcon,
  BarChart3,
  ArrowLeft,
  AlertCircle,
  Bot
} from "lucide-react";

// Color palette for charts
const COLORS = [
  "#00ADB5",
  "#FF6B6B",
  "#4ECDC4",
  "#FFE66D",
  "#95E1D3",
  "#F38181",
  "#AA96DA",
  "#FCBAD3",
  "#A8E6CF",
  "#FFD3B6",
  "#FFAAA5",
  "#FF8B94",
];

// Type definitions
type PeriodType = "all" | "30" | "90" | "180";

interface PeriodOption {
  label: string;
  value: PeriodType;
}

interface CategoryData {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface MonthData {
  month: string;
  sent: number;
  received: number;
  net: number;
}

interface DailyData {
  date: string;
  amount: number;
}

interface MerchantData {
  name: string;
  amount: number;
}

export default function DashboardPage() {
  const { transactions } = useTransactions();
  const router = useRouter();
  const [selectedPeriod, setSelectedPeriod] = useState<PeriodType>("all");

  // Redirect if no transactions
  useEffect(() => {
    if (transactions.length === 0) {
      router.push("/upload-activity");
    }
  }, [transactions.length, router]);

  // Filter transactions by period
  const filteredTransactions = useMemo(() => {
    if (selectedPeriod === "all") return transactions;
    
    const daysAgo = new Date();
    daysAgo.setDate(daysAgo.getDate() - parseInt(selectedPeriod));
    
    return transactions.filter(t => new Date(t.date) >= daysAgo);
  }, [transactions, selectedPeriod]);

  // Calculate spending by category
  const spendingByCategory = useMemo((): CategoryData[] => {
    const categoryMap: { [key: string]: number } = {};
    
    filteredTransactions
      .filter(t => t.type === "Paid" || t.type === "Sent")
      .forEach(t => {
        const category = t.category || "Miscellaneous";
        categoryMap[category] = (categoryMap[category] || 0) + t.amount;
      });

    return Object.entries(categoryMap)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [filteredTransactions]);

  // Calculate income by category
  const incomeByCategory = useMemo((): CategoryData[] => {
    const categoryMap: { [key: string]: number } = {};
    
    filteredTransactions
      .filter(t => t.type === "Received")
      .forEach(t => {
        const category = t.category || "Miscellaneous";
        categoryMap[category] = (categoryMap[category] || 0) + t.amount;
      });

    return Object.entries(categoryMap)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [filteredTransactions]);

  // Calculate monthly trend
  const monthlyTrend = useMemo((): MonthData[] => {
    const monthMap: { [key: string]: { sent: number; received: number } } = {};
    
    filteredTransactions.forEach(t => {
      const month = t.date.substring(0, 7);
      if (!monthMap[month]) {
        monthMap[month] = { sent: 0, received: 0 };
      }
      
      if (t.type === "Received") {
        monthMap[month].received += t.amount;
      } else {
        monthMap[month].sent += t.amount;
      }
    });

    return Object.entries(monthMap)
      .map(([month, data]) => ({
        month,
        sent: data.sent,
        received: data.received,
        net: data.received - data.sent,
      }))
      .sort((a, b) => a.month.localeCompare(b.month));
  }, [filteredTransactions]);

  // Calculate daily spending trend (last 30 days)
  const dailySpending = useMemo((): DailyData[] => {
    const last30Days = filteredTransactions.filter(t => {
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
      return new Date(t.date) >= thirtyDaysAgo;
    });

    const dayMap: { [key: string]: number } = {};
    
    last30Days
      .filter(t => t.type === "Paid" || t.type === "Sent")
      .forEach(t => {
        dayMap[t.date] = (dayMap[t.date] || 0) + t.amount;
      });

    return Object.entries(dayMap)
      .map(([date, amount]) => ({ date, amount }))
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-30);
  }, [filteredTransactions]);

  // Calculate top merchants
  const topMerchants = useMemo((): MerchantData[] => {
    const merchantMap: { [key: string]: number } = {};
    
    filteredTransactions
      .filter(t => t.type === "Paid" || t.type === "Sent")
      .forEach(t => {
        const merchant = t.receiver || "Unknown";
        merchantMap[merchant] = (merchantMap[merchant] || 0) + t.amount;
      });

    return Object.entries(merchantMap)
      .map(([name, amount]) => ({ name, amount }))
      .sort((a, b) => b.amount - a.amount)
      .slice(0, 10);
  }, [filteredTransactions]);

  // Calculate summary statistics
  const stats = useMemo(() => {
    const sent = filteredTransactions
      .filter(t => t.type === "Paid" || t.type === "Sent")
      .reduce((sum, t) => sum + t.amount, 0);
    
    const received = filteredTransactions
      .filter(t => t.type === "Received")
      .reduce((sum, t) => sum + t.amount, 0);
    
    const sentTransactions = filteredTransactions.filter(t => t.type === "Paid" || t.type === "Sent");
    const avgTransaction = sentTransactions.length > 0 ? sent / sentTransactions.length : 0;
    
    const sentAmounts = sentTransactions.map(t => t.amount);
    const largestExpense = sentAmounts.length > 0 ? Math.max(...sentAmounts) : 0;

    return {
      totalSent: sent,
      totalReceived: received,
      netFlow: received - sent,
      avgTransaction,
      largestExpense,
      transactionCount: filteredTransactions.length,
    };
  }, [filteredTransactions]);

  // Period options
  const periodOptions: PeriodOption[] = [
    { label: "All Time", value: "all" },
    { label: "Last 30 Days", value: "30" },
    { label: "Last 90 Days", value: "90" },
    { label: "Last 180 Days", value: "180" },
  ];

  // Show loading state while redirecting
  if (transactions.length === 0) {
    return (
      <div className="bg-[#222831] text-[#EEEEEE] min-h-screen font-sans flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl mb-4">No transactions found...</p>
          <p className="text-gray-400">Redirecting to upload page...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#222831] text-[#EEEEEE] min-h-screen font-sans">
      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="mb-8">
          <button
            onClick={() => router.push("/transactions")}
            className="flex items-center text-[#00ADB5] hover:text-white transition-colors mb-4"
          >
            <ArrowLeft className="mr-2 h-5 w-5" />
            Back to Transactions
          </button>
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <h1 className="text-4xl md:text-5xl font-bold text-white">
              Financial Dashboard
            </h1>
            <div className="flex flex-wrap gap-2">
              {periodOptions.map(period => (
                <button
                  key={period.value}
                  onClick={() => setSelectedPeriod(period.value)}
                  className={`px-4 py-2 text-sm font-bold rounded-lg transition-colors ${
                    selectedPeriod === period.value
                      ? "bg-[#00ADB5] text-white"
                      : "bg-[#393E46] hover:bg-[#4a505a] text-gray-300"
                  }`}
                >
                  {period.label}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-gray-400 text-sm font-bold uppercase">Total Spent</h3>
              <TrendingDown className="text-red-400 h-6 w-6" />
            </div>
            <p className="text-3xl font-bold text-red-400">
              ₹{stats.totalSent.toFixed(2)}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {filteredTransactions.filter(t => t.type === "Paid" || t.type === "Sent").length} transactions
            </p>
          </div>

          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-gray-400 text-sm font-bold uppercase">Total Received</h3>
              <TrendingUp className="text-green-400 h-6 w-6" />
            </div>
            <p className="text-3xl font-bold text-green-400">
              ₹{stats.totalReceived.toFixed(2)}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {filteredTransactions.filter(t => t.type === "Received").length} transactions
            </p>
          </div>

          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-gray-400 text-sm font-bold uppercase">Net Flow</h3>
              <DollarSign className={`h-6 w-6 ${stats.netFlow >= 0 ? "text-green-400" : "text-red-400"}`} />
            </div>
            <p className={`text-3xl font-bold ${stats.netFlow >= 0 ? "text-green-400" : "text-red-400"}`}>
              {stats.netFlow < 0 && "-"}₹{Math.abs(stats.netFlow).toFixed(2)}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {stats.netFlow >= 0 ? "Surplus" : "Deficit"}
            </p>
          </div>

          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-gray-400 text-sm font-bold uppercase">Avg Transaction</h3>
              <Calendar className="text-[#00ADB5] h-6 w-6" />
            </div>
            <p className="text-3xl font-bold text-[#00ADB5]">
              ₹{stats.avgTransaction.toFixed(2)}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Largest: ₹{stats.largestExpense.toFixed(2)}
            </p>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Spending by Category - Pie Chart */}
          {spendingByCategory.length > 0 && (
            <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <PieChartIcon className="mr-2 h-5 w-5 text-[#00ADB5]" />
                Spending by Category
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                    <Pie
                    data={spendingByCategory}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      `${name ?? "Other"} ${Math.round(((Number(percent) || 0) * 100))}%`
                    }
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    >
                    {spendingByCategory.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                    </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#393E46",
                      border: "none",
                      borderRadius: "8px",
                      color: "#EEEEEE",
                    }}
                    formatter={(value: number) => `₹${value.toFixed(2)}`}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}


          {/* Monthly Trend - Line Chart */}
          {monthlyTrend.length > 0 && (
            <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <TrendingUp className="mr-2 h-5 w-5 text-[#00ADB5]" />
                Monthly Cash Flow
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={monthlyTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#4a505a" />
                  <XAxis dataKey="month" stroke="#EEEEEE" />
                  <YAxis stroke="#EEEEEE" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#393E46",
                      border: "none",
                      borderRadius: "8px",
                      color: "#EEEEEE",
                    }}
                    formatter={(value: number) => `₹${value.toFixed(2)}`}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="sent" stroke="#FF6B6B" name="Spent" strokeWidth={2} />
                  <Line type="monotone" dataKey="received" stroke="#4ECDC4" name="Received" strokeWidth={2} />
                  <Line type="monotone" dataKey="net" stroke="#00ADB5" name="Net" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Top Categories - Bar Chart */}
          {spendingByCategory.length > 0 && (
            <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <BarChart3 className="mr-2 h-5 w-5 text-[#00ADB5]" />
                Top Spending Categories
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={spendingByCategory.slice(0, 8)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#4a505a" />
                  <XAxis dataKey="name" stroke="#EEEEEE" angle={-45} textAnchor="end" height={100} />
                  <YAxis stroke="#EEEEEE" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#393E46",
                      border: "none",
                      borderRadius: "8px",
                      color: "#EEEEEE",
                    }}
                    formatter={(value: number) => `₹${value.toFixed(2)}`}
                  />
                  <Bar dataKey="value" fill="#00ADB5" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Daily Spending Trend - Area Chart */}
          {dailySpending.length > 0 && (
            <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <Calendar className="mr-2 h-5 w-5 text-[#00ADB5]" />
                Daily Spending (Last 30 Days)
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={dailySpending}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#4a505a" />
                  <XAxis dataKey="date" stroke="#EEEEEE" />
                  <YAxis stroke="#EEEEEE" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#393E46",
                      border: "none",
                      borderRadius: "8px",
                      color: "#EEEEEE",
                    }}
                    formatter={(value: number) => `₹${value.toFixed(2)}`}
                  />
                  <Area
                    type="monotone"
                    dataKey="amount"
                    stroke="#00ADB5"
                    fill="#00ADB5"
                    fillOpacity={0.6}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Top Merchants Table */}
        {topMerchants.length > 0 && (
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg mb-8">
            <h3 className="text-xl font-bold text-white mb-4 flex items-center">
              <AlertCircle className="mr-2 h-5 w-5 text-[#00ADB5]" />
              Top 10 Merchants
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-[#222831]">
                  <tr>
                    <th className="p-3 text-left">#</th>
                    <th className="p-3 text-left">Merchant</th>
                    <th className="p-3 text-right">Total Spent</th>
                    <th className="p-3 text-right">% of Total</th>
                  </tr>
                </thead>
                <tbody>
                  {topMerchants.map((merchant, index) => (
                    <tr key={index} className="border-b border-gray-700 hover:bg-[#4a505a] transition-colors">
                      <td className="p-3">{index + 1}</td>
                      <td className="p-3 font-medium">{merchant.name}</td>
                      <td className="p-3 text-right font-mono">₹{merchant.amount.toFixed(2)}</td>
                      <td className="p-3 text-right text-gray-400">
                        {((merchant.amount / stats.totalSent) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Income Breakdown */}
        {incomeByCategory.length > 0 && (
          <div className="bg-[#393E46] p-6 rounded-xl shadow-lg">
            <h3 className="text-xl font-bold text-white mb-4 flex items-center">
              <TrendingUp className="mr-2 h-5 w-5 text-green-400" />
              Income Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={incomeByCategory} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#4a505a" />
                <XAxis type="number" stroke="#EEEEEE" />
                <YAxis dataKey="name" type="category" stroke="#EEEEEE" width={150} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#393E46",
                    border: "none",
                    borderRadius: "8px",
                    color: "#EEEEEE",
                  }}
                  formatter={(value: number) => `₹${value.toFixed(2)}`}
                />
                <Bar dataKey="value" fill="#4ECDC4" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </main>
      <button
        onClick={() => router.push("/advisor")}
        className="fixed bottom-8 right-8 bg-[#00ADB5] text-white p-4 rounded-full shadow-lg hover:bg-[#008a90] transition-all hover:scale-110 flex items-center gap-2 z-50"
        title="Ask AI Financial Advisor"
      >
        <Bot className="h-6 w-6" />
        <span className="hidden md:inline font-bold">AI Advisor</span>
      </button>
    </div>
  );
}
