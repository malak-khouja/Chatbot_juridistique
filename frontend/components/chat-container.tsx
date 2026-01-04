"use client"

import { useState, useEffect } from "react"
import { ExampleQuestions } from "./example-questions"
import { MessageList } from "./message-list"
import { ChatInput } from "./chat-input"
import { Card } from "./ui/card"
import { Button } from "./ui/button"
import { Scale, Moon, Sun, Menu, X } from "lucide-react"
import { ConversationHistory, type Conversation } from "./conversation-history"

export interface Message {
  id: string
  text: string
  sender: "user" | "bot"
  timestamp: Date
}

interface ConversationData {
  id: string
  messages: Message[]
  title: string
}

export function ChatContainer() {
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [conversations, setConversations] = useState<ConversationData[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showExamples, setShowExamples] = useState(true)

  useEffect(() => {
    const savedDarkMode = localStorage.getItem("darkMode")
    if (savedDarkMode === "true") {
      setIsDarkMode(true)
      document.documentElement.classList.add("dark")
    }
  }, [])

  const toggleDarkMode = () => {
    setIsDarkMode((prev) => {
      const newValue = !prev
      localStorage.setItem("darkMode", newValue.toString())
      if (newValue) {
        document.documentElement.classList.add("dark")
      } else {
        document.documentElement.classList.remove("dark")
      }
      return newValue
    })
  }

  const createNewConversation = () => {
    const newConversation: ConversationData = {
      id: Date.now().toString(),
      messages: [],
      title: "Nouvelle conversation",
    }
    setConversations((prev) => [newConversation, ...prev])
    setCurrentConversationId(newConversation.id)
    setMessages([])
    setIsSidebarOpen(false)
  }

  const selectConversation = (id: string) => {
    const conversation = conversations.find((c) => c.id === id)
    if (conversation) {
      setCurrentConversationId(id)
      setMessages(conversation.messages)
      setIsSidebarOpen(false)
    }
  }

  const deleteConversation = (id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id))
    if (currentConversationId === id) {
      setCurrentConversationId(null)
      setMessages([])
    }
  }

  useEffect(() => {
    if (currentConversationId && messages.length > 0) {
      setConversations((prev) =>
        prev.map((conv) => {
          if (conv.id === currentConversationId) {
            const firstUserMessage = messages.find((m) => m.sender === "user")
            return {
              ...conv,
              messages,
              title: firstUserMessage ? firstUserMessage.text.slice(0, 50) + "..." : conv.title,
            }
          }
          return conv
        }),
      )
    }
  }, [messages, currentConversationId])

  const sendMessage = async (text: string) => {
    // Hide examples immediately when user sends message
    setShowExamples(false)

    if (!currentConversationId) {
      createNewConversation()
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: "user",
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: text }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response")
      }

      const data = await response.json()

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.answer || "Je suis désolé, je n'ai pas pu traiter votre demande.",
        sender: "bot",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, botMessage])
      setShowExamples(true)
    } catch (error) {
      console.error("Error sending message:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Une erreur s'est produite. Veuillez vérifier que votre serveur backend est démarré sur http://localhost:8000",
        sender: "bot",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
      setShowExamples(true)
    } finally {
      setIsLoading(false)
    }
  }

  const conversationList: Conversation[] = conversations.map((conv) => ({
    id: conv.id,
    title: conv.title,
    lastMessage: conv.messages[conv.messages.length - 1]?.text || "Aucun message",
    timestamp: conv.messages[conv.messages.length - 1]?.timestamp || new Date(),
  }))

  return (
    <div className="flex min-h-screen bg-legal-background">
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 transform transition-transform duration-300 md:w-72 lg:static lg:translate-x-0 ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <ConversationHistory
          conversations={conversationList}
          currentConversationId={currentConversationId}
          onSelectConversation={selectConversation}
          onNewConversation={createNewConversation}
          onDeleteConversation={deleteConversation}
        />
      </aside>

      {isSidebarOpen && (
        <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setIsSidebarOpen(false)} />
      )}

      {/* Main Content */}
      <div className="flex min-h-screen flex-1 flex-col">
        {/* Header */}
        <header className="border-b border-legal-border bg-legal-surface/50 backdrop-blur-sm">
          <div className="container mx-auto flex items-center gap-2 px-3 py-3 sm:gap-3 sm:px-4 sm:py-4">
            <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
              {isSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>

            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-legal-primary sm:h-10 sm:w-10">
              <Scale className="h-4 w-4 text-white sm:h-5 sm:w-5" />
            </div>
            <div className="flex-1 min-w-0">
              <h1 className="truncate text-base font-semibold text-legal-text sm:text-lg">Assistant Juridique IA</h1>
              <p className="hidden text-sm text-legal-muted sm:block">Votre conseiller juridique intelligent</p>
            </div>

            <Button variant="ghost" size="icon" onClick={toggleDarkMode} className="shrink-0">
              {isDarkMode ? (
                <Sun className="h-5 w-5 text-legal-primary" />
              ) : (
                <Moon className="h-5 w-5 text-legal-primary" />
              )}
            </Button>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="container mx-auto flex max-w-4xl flex-1 flex-col px-3 py-4 sm:px-4 sm:py-6">
            {/* Legal Disclaimer */}
            <Card className="mb-4 border-legal-warning/20 bg-legal-warning/5 p-3 sm:mb-6 sm:p-4">
              <p className="text-balance text-xs text-legal-text/80 sm:text-sm">
                ⚖️ <strong>Avertissement :</strong> Les réponses fournies ne remplacent pas un avis juridique
                professionnel. Consultez toujours un avocat qualifié pour des conseils juridiques spécifiques à votre
                situation.
              </p>
            </Card>

            {showExamples && !isLoading && (
              <div className="mb-4 sm:mb-6">
                <h2 className="mb-3 text-center text-sm font-medium text-legal-muted sm:mb-4 sm:text-base">
                  {messages.length === 0
                    ? "Posez votre question ou choisissez un exemple"
                    : "Choisissez une autre question"}
                </h2>
                <ExampleQuestions onSelectQuestion={sendMessage} />
              </div>
            )}

            {messages.length > 0 && (
              <div className="mb-4 flex-1 overflow-hidden sm:mb-6">
                <MessageList messages={messages} isLoading={isLoading} />
              </div>
            )}

            {/* Input */}
            <ChatInput onSendMessage={sendMessage} disabled={isLoading} />
          </div>
        </div>
      </div>
    </div>
  )
}
