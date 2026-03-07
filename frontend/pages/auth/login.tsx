import { useState } from "react"
import { supabase } from "../../lib/supabaseClient"
import Link from "next/link"
import { useRouter } from "next/router"

export default function LoginPage() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    const { error } = await supabase.auth.signInWithPassword({ email, password })

    if (error) {
      setError(error.message)
      setIsLoading(false)
    } else {
      router.push("/")
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.backgroundGlow} />

      <div style={styles.card}>
        <div style={styles.logoSection}>
          <div style={styles.icon}>🙏</div>
          <h1 style={styles.title}>VedAI</h1>
          <p style={styles.subtitle}>Welcome back, seeker of wisdom</p>
        </div>

        <form onSubmit={handleLogin} style={styles.form}>
          <div style={styles.inputWrapper}>
            <label style={styles.label}>Email Address</label>
            <input 
              style={styles.input}
              type="email"
              value={email} 
              onChange={(e) => setEmail(e.target.value)} 
              placeholder="Enter your email" 
              required
            />
          </div>
          
          <div style={styles.inputWrapper}>
            <label style={styles.label}>Password</label>
            <input 
              style={styles.input}
              type="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              placeholder="Enter your password" 
              required
            />
          </div>

          <button 
            type="submit" 
            style={{
              ...styles.button,
              opacity: isLoading ? 0.7 : 1,
              cursor: isLoading ? "not-allowed" : "pointer"
            }} 
            disabled={isLoading}
          >
            {isLoading ? "Signing in..." : "Sign In"}
          </button>
        </form>

        {error && (
          <div style={styles.errorContainer}>
            <span style={styles.errorIcon}>⚠️</span>
            <p style={styles.errorText}>{error}</p>
          </div>
        )}

        <div style={styles.divider}>
          <div style={styles.line} />
          <span style={styles.dividerText}>or</span>
          <div style={styles.line} />
        </div>

        <p style={styles.footer}>
          New to the journey? <Link href="/auth/register" style={styles.link}>Create an account</Link>
        </p>
      </div>

      <style jsx global>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
          background-color: #0f0f0f;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        input:focus {
          border-color: #f59e0b !important;
          box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2);
        }
        button:hover:not(:disabled) {
          background-color: #fbbf24 !important;
          transform: translateY(-1px);
        }
        button:active:not(:disabled) {
          transform: translateY(0);
        }
      `}</style>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    minHeight: "100vh",
    backgroundColor: "#0f0f0f",
    color: "#e0e0e0",
    position: "relative",
    overflow: "hidden",
  },
  backgroundGlow: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: "600px",
    height: "600px",
    background: "radial-gradient(circle, rgba(245, 158, 11, 0.05) 0%, rgba(0,0,0,0) 70%)",
    pointerEvents: "none",
  },
  card: {
    backgroundColor: "rgba(26, 26, 26, 0.8)",
    backdropFilter: "blur(12px)",
    padding: "48px 40px",
    borderRadius: "24px",
    border: "1px solid #333",
    width: "100%",
    maxWidth: "440px",
    boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4)",
    zIndex: 1,
  },
  logoSection: {
    textAlign: "center",
    marginBottom: "32px",
  },
  icon: {
    fontSize: "40px",
    marginBottom: "12px",
  },
  title: {
    fontSize: "32px",
    fontWeight: "700",
    color: "#f59e0b",
    letterSpacing: "-0.025em",
    marginBottom: "8px",
  },
  subtitle: {
    fontSize: "16px",
    color: "#888",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  inputWrapper: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  label: {
    fontSize: "14px",
    fontWeight: "500",
    color: "#aaa",
    marginLeft: "4px",
  },
  input: {
    width: "100%",
    padding: "14px 16px",
    backgroundColor: "#262626",
    border: "1px solid #333",
    borderRadius: "12px",
    color: "#fff",
    fontSize: "16px",
    outline: "none",
    transition: "all 0.2s",
  },
  button: {
    width: "100%",
    padding: "14px",
    backgroundColor: "#f59e0b",
    color: "#000",
    border: "none",
    borderRadius: "12px",
    fontWeight: "600",
    fontSize: "16px",
    marginTop: "12px",
    transition: "all 0.2s",
  },
  errorContainer: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    backgroundColor: "rgba(239, 68, 68, 0.1)",
    border: "1px solid rgba(239, 68, 68, 0.2)",
    padding: "12px",
    borderRadius: "12px",
    marginTop: "20px",
  },
  errorIcon: {
    fontSize: "14px",
  },
  errorText: {
    color: "#ef4444",
    fontSize: "14px",
    fontWeight: "500",
  },
  divider: {
    display: "flex",
    alignItems: "center",
    gap: "16px",
    margin: "32px 0",
  },
  line: {
    flex: 1,
    height: "1px",
    backgroundColor: "#333",
  },
  dividerText: {
    color: "#666",
    fontSize: "14px",
    textTransform: "lowercase",
  },
  footer: {
    textAlign: "center",
    fontSize: "15px",
    color: "#888",
  },
  link: {
    color: "#f59e0b",
    textDecoration: "none",
    fontWeight: "600",
    marginLeft: "4px",
  }
}


