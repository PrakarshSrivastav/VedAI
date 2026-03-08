import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/router"
import { supabase } from "../lib/supabaseClient"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  context?: { chapter_number: string; chapter_verse: string; translation: string }[]
}

// Mock responses for demo (replace with API call later)
const mockResponses: Record<string, { answer: string; context: { chapter_number: string; chapter_verse: string; translation: string }[] }> = {
  default: {
    answer: "Based on the teachings of the Bhagavad Gita, one should perform their duty without attachment to the results. Krishna emphasizes that action performed with wisdom and detachment leads to liberation. The wise see inaction in action and action in inaction.",
    context: [
      { chapter_number: "2", chapter_verse: "2.47", translation: "You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions." },
      { chapter_number: "3", chapter_verse: "3.19", translation: "Therefore, without attachment, perform always the work that has to be done, for man attains to the highest by doing work without attachment." },
      { chapter_number: "4", chapter_verse: "4.18", translation: "One who sees inaction in action and action in inaction is wise among men." },
    ]
  }
}

async function queryBackend(question: string): Promise<{ answer: string; context: Message["context"] }> {
  // Try real backend first
  try {
    const { data: { session } } = await supabase.auth.getSession()
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    }
    
    if (session?.access_token) {
      headers["Authorization"] = `Bearer ${session.access_token}`
    }

    const res = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers,
      body: JSON.stringify({ question })
    })
    if (res.ok) {
      return await res.json()
    }
  } catch (e) {
    console.log("Backend unavailable, using mock data")
  }

  // Fallback to mock
  await new Promise(r => setTimeout(r, 1000)) // Simulate delay
  return mockResponses.default
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isCheckingAuth, setIsCheckingAuth] = useState(true)
  const [session, setSession] = useState<any>(null)
  const [freePromptsUsed, setFreePromptsUsed] = useState(0)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const router = useRouter()

  useEffect(() => {
    const checkUser = async () => {
      const { data: { session: currentSession } } = await supabase.auth.getSession()
      setSession(currentSession)
      setIsCheckingAuth(false)
    }
    
    checkUser()

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
    })

    return () => subscription.unsubscribe()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    // Auth & Rate Limit Check
    if (!session && freePromptsUsed >= 1) {
      const shouldLogin = window.confirm("You've used your free prompt. Please log in to continue your journey of wisdom.")
      if (shouldLogin) {
        router.push("/auth/login")
      }
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    if (!session) {
      setFreePromptsUsed(prev => prev + 1)
    }

    try {
      const response = await queryBackend(userMessage.content)
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        context: response.context
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again."
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  if (isCheckingAuth) {
    return (
      <div style={styles.loadingContainer}>
        <div style={styles.loadingDots}>
          <span>●</span><span>●</span><span>●</span>
        </div>
      </div>
    )
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div>
            <h1 style={styles.title}>VedAI</h1>
            <p style={styles.subtitle}>Ask questions about the Bhagavad Gita</p>
          </div>
          {session ? (
            <button 
              style={styles.logoutBtn}
              onClick={async () => {
                await supabase.auth.signOut()
                router.push("/auth/login")
              }}
            >
              Logout
            </button>
          ) : (
            <button 
              style={styles.loginBtn}
              onClick={() => router.push("/auth/login")}
            >
              Sign In
            </button>
          )}
        </div>
      </header>

      <main style={styles.chatContainer}>
        {messages.length === 0 && (
          <div style={styles.emptyState}>
            <div style={styles.emptyIcon}>🙏</div>
            <h2 style={styles.emptyTitle}>Welcome to VedAI</h2>
            {!session && (
              <p style={styles.freePromptBadge}>Try one free prompt without an account</p>
            )}
            <p style={styles.emptyText}>
              Ask any question about the Bhagavad Gita and receive wisdom from the sacred text.
            </p>
            <div style={styles.suggestions}>
              <p style={styles.suggestionsTitle}>Try asking:</p>
              {[
                "What does Krishna say about duty?",
                "How can I find inner peace?",
                "What is the meaning of karma?",
              ].map((suggestion, i) => (
                <button
                  key={i}
                  style={styles.suggestionBtn}
                  onClick={() => setInput(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              ...styles.message,
              ...(msg.role === "user" ? styles.userMessage : styles.assistantMessage)
            }}
          >
            <div style={styles.messageRole}>
              {msg.role === "user" ? "You" : "VedAI"}
            </div>
            <div style={styles.messageContent}>{msg.content}</div>

            {msg.context && msg.context.length > 0 && (
              <div style={styles.contextSection}>
                <div style={styles.contextTitle}>Referenced Verses:</div>
                {msg.context.map((verse, i) => (
                  <div key={i} style={styles.verse}>
                    <span style={styles.verseRef}>
                      Chapter {verse.chapter_number}, Verse {verse.chapter_verse}
                    </span>
                    <p style={styles.verseText}>{verse.translation}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div style={{ ...styles.message, ...styles.assistantMessage }}>
            <div style={styles.messageRole}>VedAI</div>
            <div className="loading-dots" style={styles.loadingDots}>
              <span>●</span><span>●</span><span>●</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      <form onSubmit={handleSubmit} style={styles.inputForm}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the Bhagavad Gita..."
          style={styles.input}
          disabled={isLoading}
        />
        <button
          type="submit"
          style={{
            ...styles.sendBtn,
            opacity: isLoading || !input.trim() ? 0.5 : 1
          }}
          disabled={isLoading || !input.trim()}
        >
          Send
        </button>
      </form>

      <style jsx global>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        .loading-dots span:nth-child(1) { animation: pulse 1.4s infinite 0s; }
        .loading-dots span:nth-child(2) { animation: pulse 1.4s infinite 0.2s; }
        .loading-dots span:nth-child(3) { animation: pulse 1.4s infinite 0.4s; }
        input:focus { border-color: #f59e0b !important; }
        button:hover:not(:disabled) { transform: scale(1.02); }
      `}</style>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    backgroundColor: "#0f0f0f",
    color: "#e0e0e0",
  },
  header: {
    padding: "16px 24px",
    borderBottom: "1px solid #2a2a2a",
  },
  headerContent: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    maxWidth: "1200px",
    margin: "0 auto",
    width: "100%",
  },
  title: {
    fontSize: "24px",
    fontWeight: 600,
    color: "#f59e0b",
    margin: 0,
  },
  subtitle: {
    fontSize: "14px",
    color: "#888",
    marginTop: "4px",
  },
  logoutBtn: {
    padding: "8px 16px",
    backgroundColor: "transparent",
    color: "#888",
    border: "1px solid #333",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
  },
  loginBtn: {
    padding: "8px 20px",
    backgroundColor: "#f59e0b",
    color: "#000",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: "600",
  },
  freePromptBadge: {
    backgroundColor: "rgba(245, 158, 11, 0.1)",
    color: "#f59e0b",
    padding: "4px 12px",
    borderRadius: "16px",
    fontSize: "12px",
    fontWeight: "500",
    marginBottom: "16px",
    border: "1px solid rgba(245, 158, 11, 0.2)",
  },
  chatContainer: {
    flex: 1,
    overflowY: "auto",
    padding: "24px",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    textAlign: "center",
    padding: "24px",
  },
  emptyIcon: {
    fontSize: "48px",
    marginBottom: "16px",
  },
  emptyTitle: {
    fontSize: "24px",
    fontWeight: 600,
    color: "#f59e0b",
    marginBottom: "8px",
  },
  emptyText: {
    fontSize: "16px",
    color: "#888",
    maxWidth: "400px",
    marginBottom: "24px",
  },
  suggestions: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    alignItems: "center",
  },
  suggestionsTitle: {
    fontSize: "14px",
    color: "#666",
    marginBottom: "8px",
  },
  suggestionBtn: {
    padding: "10px 16px",
    backgroundColor: "#1a1a1a",
    border: "1px solid #333",
    borderRadius: "8px",
    color: "#e0e0e0",
    cursor: "pointer",
    fontSize: "14px",
    transition: "all 0.2s",
  },
  message: {
    maxWidth: "800px",
    padding: "16px",
    borderRadius: "12px",
  },
  userMessage: {
    backgroundColor: "#1e3a5f",
    alignSelf: "flex-end",
    marginLeft: "auto",
  },
  assistantMessage: {
    backgroundColor: "#1a1a1a",
    alignSelf: "flex-start",
    border: "1px solid #2a2a2a",
  },
  messageRole: {
    fontSize: "12px",
    fontWeight: 600,
    color: "#f59e0b",
    marginBottom: "8px",
    textTransform: "uppercase",
  },
  messageContent: {
    fontSize: "15px",
    lineHeight: 1.6,
  },
  contextSection: {
    marginTop: "16px",
    paddingTop: "16px",
    borderTop: "1px solid #333",
  },
  contextTitle: {
    fontSize: "12px",
    fontWeight: 600,
    color: "#888",
    marginBottom: "12px",
    textTransform: "uppercase",
  },
  verse: {
    marginBottom: "12px",
    paddingLeft: "12px",
    borderLeft: "2px solid #f59e0b",
  },
  verseRef: {
    fontSize: "12px",
    color: "#f59e0b",
    fontWeight: 500,
  },
  verseText: {
    fontSize: "14px",
    color: "#aaa",
    marginTop: "4px",
    fontStyle: "italic",
  },
  loadingDots: {
    display: "flex",
    gap: "4px",
    fontSize: "20px",
    color: "#f59e0b",
  },
  loadingContainer: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "#0f0f0f",
  },
  inputForm: {
    display: "flex",
    gap: "12px",
    padding: "16px 24px",
    borderTop: "1px solid #2a2a2a",
    backgroundColor: "#0f0f0f",
  },
  input: {
    flex: 1,
    padding: "14px 18px",
    fontSize: "15px",
    backgroundColor: "#1a1a1a",
    border: "1px solid #333",
    borderRadius: "12px",
    color: "#e0e0e0",
    outline: "none",
  },
  sendBtn: {
    padding: "14px 24px",
    fontSize: "15px",
    fontWeight: 600,
    backgroundColor: "#f59e0b",
    color: "#000",
    border: "none",
    borderRadius: "12px",
    cursor: "pointer",
  },
}

