import * as React from "react";
import { LuBot, LuSendHorizontal, LuUpload } from "react-icons/lu";
import useChatbot from "../hooks/useChatbot";
import Markdown from "react-markdown";
import useChatScroll from "../hooks/useChatScroll";
import { T } from "../libs/type";

const ChatComponent: React.FunctionComponent = () => {
  const [input, setInput] = React.useState("");
  const [file, setFile] = React.useState<File | null>(null);

  const { messages, sendMessage } = useChatbot();
  const ref = useChatScroll(messages);

  const handleSend = () => {
    if (input.trim()) {
      sendMessage(input);
      console.log("Attached file:", file);
      setInput("");
      setFile(null);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 px-4">
      <div className="w-full max-w-4xl h-[85vh] flex flex-col bg-white rounded-2xl shadow-xl overflow-hidden">
        {/* Header */}
        <h2 className="p-4 font-medium text-lg text-center bg-gradient-to-r from-blue-500 to-indigo-600 text-white flex items-center justify-center gap-2">
          Team E | 7 | MidTerm Project <LuBot size={24} />
        </h2>

        {/* Chat Area */}
        <div
          ref={ref}
          className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50"
        >
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`p-4 rounded-2xl max-w-md ${
                  msg.sender === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-white text-gray-800 shadow-md"
                }`}
              >
                <Markdown>{msg.text}</Markdown>
              </div>
            </div>
          ))}

          {file && (
            <div className="text-sm text-gray-500 flex justify-end">
              📎 Attached file: {file.name}
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 bg-white border-t border-gray-200">
          <div className="flex items-center gap-3 max-w-3xl mx-auto">
            <input
              type="text"
              className="flex-1 p-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50 text-gray-800 placeholder-gray-400"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e: T) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />

            {/* Upload Button */}
            <label className="cursor-pointer p-2 text-gray-500 hover:text-blue-600 transition-colors">
              <LuUpload size={22} />
              <input
                type="file"
                accept=".txt, .pdf"
                className="hidden"
                onChange={(e) => {
                  const selectedFile = e.target.files?.[0];
                  if (selectedFile) {
                    setFile(selectedFile);
                    console.log("Uploaded file:", selectedFile);
                  }
                }}
              />
            </label>

            {/* Send Button */}
            <button
              onClick={handleSend}
              className="p-2 text-white bg-blue-600 rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-50"
              disabled={!input.trim()}
            >
              <LuSendHorizontal size={22} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatComponent;