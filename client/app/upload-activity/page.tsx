"use client";

import { useState, DragEvent } from "react";
import { CheckCircle, UploadCloud, File, X } from "lucide-react";

// Main Page Component
export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (files: FileList | null) => {
    setError(null);
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type === "text/html") {
        setSelectedFile(file);
      } else {
        setError("Invalid file type. Please upload an HTML file.");
        setSelectedFile(null);
      }
    }
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileChange(e.dataTransfer.files);
  };

  const removeFile = () => {
    setSelectedFile(null);
    setError(null);
  };

  const steps = [
    {
      text: "Go to",
      link: { href: "https://takeout.google.com/", label: "Google Takeout" },
    },
    { text: 'Click "Deselect all" products.' },
    {
      text: 'Press <kbd>Ctrl</kbd> + <kbd>F</kbd>, search for and select "Google Pay".',
    },
    {
      text: 'Under Google Pay, click "All Google Pay data included". In the pop-up, click "Deselect all" and then select only "My Activity".',
    },
    { text: 'Scroll to the bottom and click "Next step".' },
    {
      text: 'Choose "Send download link via email" and click "Create export".',
    },
    { text: "Download the .zip file from the link you receive in your email." },
    {
      text: 'Unzip the file to find the "MyActivity.html" file, and upload it here.',
    },
  ];

  return (
    <div className="bg-[#222831] text-[#EEEEEE] min-h-screen font-sans">
      <main className="container mx-auto px-4 py-8 sm:py-16">
        <header className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-white">
            Google Pay Activity Upload
          </h1>
          <p className="text-lg text-gray-400 mt-3 max-w-3xl mx-auto">
            Follow the steps below to download your transaction history and
            upload the HTML file.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
          <div className="bg-[#393E46] p-6 sm:p-8 rounded-xl shadow-lg">
            <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
              <CheckCircle className="text-[#00ADB5] mr-3 h-7 w-7" />
              How to Download Your File
            </h2>
            <ol className="space-y-4">
              {steps.map((step, index) => (
                <li key={index} className="flex items-start">
                  <span className="flex-shrink-0 bg-[#00ADB5] text-white text-sm font-bold w-6 h-6 rounded-full flex items-center justify-center mr-4 mt-1">
                    {index + 1}
                  </span>
                  <span
                    className="text-gray-300"
                    dangerouslySetInnerHTML={{
                      __html: step.link
                        ? `${step.text} <a href="${step.link.href}" target="_blank" rel="noopener noreferrer" class="text-[#00ADB5] font-medium underline hover:text-white transition-colors">${step.link.label}</a>.`
                        : step.text
                            .replace(
                              /<kbd>/g,
                              '<kbd class="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded-md">'
                            )
                            .replace(/<\/kbd>/g, "</kbd>"),
                    }}
                  />
                </li>
              ))}
            </ol>
          </div>

          <div className="bg-[#393E46] p-6 sm:p-8 rounded-xl shadow-lg flex flex-col items-center justify-center">
            <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
              <UploadCloud className="text-[#00ADB5] mr-3 h-7 w-7" />
              Upload Your HTML File
            </h2>

            <div
              className={`w-full p-8 text-center border-2 border-dashed rounded-lg transition-colors duration-300 ${
                isDragging
                  ? "border-[#00ADB5] bg-[#4a505a]"
                  : "border-gray-600 hover:border-[#00ADB5]"
              }`}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                type="file"
                id="file-input"
                className="hidden"
                accept=".html"
                onChange={(e) => handleFileChange(e.target.files)}
              />

              {!selectedFile ? (
                <label htmlFor="file-input" className="cursor-pointer">
                  <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
                  <p className="mt-2 text-gray-300">
                    <span className="font-medium text-[#00ADB5]">
                      Click to upload
                    </span>{" "}
                    or drag and drop
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    HTML files only (e.g., MyActivity.html)
                  </p>
                </label>
              ) : (
                <div className="text-left">
                  <p className="text-lg font-medium text-white mb-4">
                    File Ready for Upload:
                  </p>
                  <div className="bg-[#222831] p-4 rounded-lg flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <File className="h-6 w-6 text-[#00ADB5]" />
                      <span className="text-sm font-medium text-gray-200 truncate">
                        {selectedFile.name}
                      </span>
                    </div>
                    <button
                      onClick={removeFile}
                      className="p-1 rounded-full hover:bg-gray-600 transition-colors"
                    >
                      <X className="h-5 w-5 text-gray-400" />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {error && <p className="mt-4 text-sm text-red-400">{error}</p>}

            <button
              disabled={!selectedFile}
              className="w-full mt-6 bg-[#00ADB5] text-white font-bold py-3 px-4 rounded-lg hover:bg-[#008a90] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 disabled:transform-none"
            >
              Process File
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
