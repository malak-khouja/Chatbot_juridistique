"use client"

import { Button } from "./ui/button"
import { ScrollArea } from "./ui/scroll-area"
import { MessageSquare, Plus, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"

export interface Conversation {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
}

interface ConversationHistoryProps {
  conversations: Conversation[]
  currentConversationId: string | null
  onSelectConversation: (id: string) => void
  onNewConversation: () => void
  onDeleteConversation: (id: string) => void
}

export function ConversationHistory({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}: ConversationHistoryProps) {
  return (
    <div className="flex h-full flex-col border-r border-legal-border bg-legal-surface">
      {/* New Conversation Button */}
      <div className="border-b border-legal-border p-2 sm:p-3">
        <Button
          onClick={onNewConversation}
          className="w-full justify-start gap-2 bg-legal-primary text-xs text-white hover:bg-legal-primary/90 sm:text-sm"
        >
          <Plus className="h-4 w-4" />
          Nouvelle conversation
        </Button>
      </div>

      {/* Conversations List */}
      <ScrollArea className="flex-1">
        <div className="space-y-1 p-1 sm:p-2">
          {conversations.length === 0 ? (
            <div className="px-2 py-6 text-center text-xs text-legal-muted sm:px-3 sm:py-8 sm:text-sm">
              Aucune conversation pour le moment
            </div>
          ) : (
            conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={cn(
                  "group relative flex cursor-pointer items-center gap-2 rounded-lg px-2 py-2 transition-colors hover:bg-legal-accent sm:gap-3 sm:px-3 sm:py-3",
                  currentConversationId === conversation.id && "bg-legal-accent",
                )}
                onClick={() => onSelectConversation(conversation.id)}
              >
                <MessageSquare className="h-3.5 w-3.5 shrink-0 text-legal-primary sm:h-4 sm:w-4" />
                <div className="flex-1 overflow-hidden min-w-0">
                  <p className="truncate text-xs font-medium text-legal-text sm:text-sm">{conversation.title}</p>
                  <p className="truncate text-xs text-legal-muted">{conversation.lastMessage}</p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 shrink-0 opacity-0 transition-opacity group-hover:opacity-100 sm:h-7 sm:w-7"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteConversation(conversation.id)
                  }}
                >
                  <Trash2 className="h-3 w-3 text-destructive sm:h-3.5 sm:w-3.5" />
                </Button>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
