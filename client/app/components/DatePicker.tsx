"use client";

import * as React from "react";
import { format } from "date-fns";
import { Calendar as CalendarIcon, X } from "lucide-react";
import { DateRange } from "react-day-picker";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";

interface DatePickerWithRangeProps {
  date: DateRange | undefined;
  onDateChange: (date: DateRange | undefined) => void;
  className?: string;
}

export function DatePickerWithRange({
  date,
  onDateChange,
  className,
}: DatePickerWithRangeProps) {
  const clearDates = () => onDateChange(undefined);

  return (
    <div className={className}>
      <Popover>
        <PopoverTrigger asChild>
          {/* BUTTON: Changed background, hover, and text colors to match the TransactionsPage theme */}
          <button className="flex items-center gap-2 rounded-full px-4 py-2 bg-[#393E46] text-white hover:bg-[#4a505a] transition">
            <CalendarIcon className="h-4 w-4 text-gray-400" />
            {date?.from ? (
              /* DATE DISPLAY: Changed badge background to primary teal accent */
              <span className="flex items-center gap-2 bg-[#00ADB5] text-white px-3 py-1 rounded-full text-sm">
                {date.to ? (
                  <>
                    {format(date.from, "MMM dd")} →{" "}
                    {format(date.to, "MMM dd, y")}
                  </>
                ) : (
                  format(date.from, "MMM dd, y")
                )}
                {/* CLEAR ICON: Changed hover color to red for negative action/contrast */}
                <X
                  className="h-3 w-3 cursor-pointer hover:text-red-300"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearDates();
                  }}
                />
              </span>
            ) : (
              <span className="text-sm text-gray-400">Select range</span>
            )}
          </button>
        </PopoverTrigger>
        <PopoverContent
          /* POPUP CONTENT: Changed background and border to match TransactionsPage theme */
          className="p-4 w-auto rounded-xl border border-[#222831] bg-[#393E46] text-gray-200 shadow-lg"
          align="start"
        >
          <div className="space-y-3">
            {/* TITLE: Changed color to primary teal accent */}
            <div className="text-sm font-semibold text-[#00ADB5]">
              Choose a date range
            </div>
            <Calendar
              initialFocus
              mode="range"
              defaultMonth={date?.from}
              selected={date}
              onSelect={onDateChange}
              numberOfMonths={2}
              /* CALENDAR BODY: Changed background and border to match TransactionsPage theme */
              className="rounded-lg border border-[#222831] bg-[#222831] text-gray-100"
            />
            {date?.from && (
              /* CLEAR BUTTON: Changed color to primary teal accent */
              <button
                onClick={clearDates}
                className="text-xs text-[#00ADB5] hover:underline self-end"
              >
                Clear selection
              </button>
            )}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
