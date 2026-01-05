"use client"

import { Button } from "./ui/button"
import { Card } from "./ui/card"
import { FileText, Briefcase, Building, UserCheck } from "lucide-react"

interface ExampleQuestionsProps {
  onSelectQuestion: (question: string) => void
}

const exampleQuestions = [
  {
    icon: Briefcase,
    question: "Quels sont mes droits en cas de licenciement ?",
    category: "Droit du travail",
  },
  {
    icon: FileText,
    question: "Comment rédiger un contrat de travail ?",
    category: "Contrats",
  },
  {
    icon: Building,
    question: "Quelles sont les étapes pour créer une entreprise ?",
    category: "Droit des affaires",
  },
  {
    icon: UserCheck,
    question: "Comment protéger mes droits de propriété intellectuelle ?",
    category: "Propriété intellectuelle",
  },
]

export function ExampleQuestions({ onSelectQuestion }: ExampleQuestionsProps) {
  return (
    <div className="grid gap-2 sm:grid-cols-2 sm:gap-3">
      {exampleQuestions.map((item, index) => {
        const Icon = item.icon
        return (
          <Card
            key={index}
            className="group cursor-pointer border-legal-border bg-legal-surface transition-all hover:border-legal-primary/40 hover:shadow-lg hover:shadow-legal-primary/5"
            onClick={() => onSelectQuestion(item.question)}
          >
            <Button
              variant="ghost"
              className="h-auto w-full justify-start gap-2 p-3 text-left transition-colors sm:gap-3 sm:p-4"
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-legal-primary/10 transition-colors group-hover:bg-legal-primary/20 sm:h-10 sm:w-10">
                <Icon className="h-4 w-4 text-legal-primary sm:h-5 sm:w-5" />
              </div>
             <div className="flex-1 space-y-1">
                <p className="text-xs font-medium text-legal-text sm:text-sm break-words whitespace-normal">{item.question}</p>
                <p className="hidden text-xs text-legal-muted sm:block truncate">{item.category}</p>
              </div>
            </Button>
          </Card>
        )
      })}
    </div>
  )
}
