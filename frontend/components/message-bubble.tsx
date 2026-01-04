import type { Message } from "./chat-container"
import { Scale } from "lucide-react"
import { cn } from "@/lib/utils"

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.sender === "user"

  return (
    <div
      className={cn(
        "flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300",
        isUser && "flex-row-reverse",
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-sm font-semibold",
          isUser ? "bg-legal-accent text-legal-accent-foreground" : "bg-legal-primary text-white",
        )}
      >
        {isUser ? "U" : <Scale className="h-4 w-4" />}
      </div>

      {/* Message Content */}
      <div
        className={cn(
          "max-w-[80%] break-words rounded-2xl px-4 py-3 shadow-sm",
          isUser
            ? "rounded-tr-sm bg-legal-accent text-legal-accent-foreground"
            : "rounded-tl-sm bg-legal-surface text-legal-text",
        )}
      >
        <p className="break-words text-pretty text-sm leading-relaxed [overflow-wrap:anywhere]">{message.text}</p>
        <span className="mt-1.5 block text-xs opacity-60">
          {message.timestamp.toLocaleTimeString("fr-FR", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </span>
      </div>
    </div>
  )
}
