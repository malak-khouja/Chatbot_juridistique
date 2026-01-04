"use client"

import type React from "react"

import { useState, useRef, type KeyboardEvent } from "react"
import { Button } from "./ui/button"
import { Textarea } from "./ui/textarea"
import { Send } from "lucide-react"

interface ChatInputProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
}

export function ChatInput({ onSendMessage, disabled }: ChatInputProps) {
  const [message, setMessage] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim())
      setMessage("")
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto"
      }
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value)
    // Auto-resize textarea
    e.target.style.height = "auto"
    e.target.style.height = `${e.target.scrollHeight}px`
  }

  return (
    <div className="border-t border-legal-border bg-legal-surface/50 p-4 backdrop-blur-sm">
      <div className="flex gap-2">
        <Textarea
          ref={textareaRef}
          value={message}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Posez votre question juridique..."
          disabled={disabled}
          className="min-h-[60px] max-h-[200px] resize-none border-legal-border bg-background text-legal-text placeholder:text-legal-muted focus-visible:ring-legal-primary"
          rows={1}
        />
        <Button
          onClick={handleSend}
          disabled={disabled || !message.trim()}
          size="icon"
          className="h-[60px] w-[60px] shrink-0 bg-legal-primary hover:bg-legal-primary/90"
        >
          <Send className="h-5 w-5" />
          <span className="sr-only">Envoyer le message</span>
        </Button>
      </div>
      <p className="mt-2 text-xs text-legal-muted">
        Appuyez sur Entrée pour envoyer, Maj+Entrée pour une nouvelle ligne
      </p>
    </div>
  )
}
