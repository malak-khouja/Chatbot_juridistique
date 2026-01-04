"use client"

import { useEffect, useRef } from "react"
import { MessageBubble } from "./message-bubble"
import type { Message } from "./chat-container"
import { Spinner } from "./ui/spinner"

interface MessageListProps {
  messages: Message[]
  isLoading: boolean
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  if (messages.length === 0) {
    return null
  }

  return (
    <div className="space-y-4">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {isLoading && (
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-legal-primary text-white">
            IA
          </div>
          <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm bg-legal-surface px-4 py-3 shadow-sm">
            <Spinner className="h-4 w-4 text-legal-primary" />
            <span className="text-sm text-legal-muted">Le chatbot écrit…</span>
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  )
}
