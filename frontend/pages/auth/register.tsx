import { useState } from "react"
import { supabase } from "../../lib/supabaseClient"

export default function RegisterPage() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")

  const handleRegister = async () => {
    const { error } = await supabase.auth.signUp({ email, password })
    if (error) setError(error.message)
  }

  return (
    <div>
      <h1>Register</h1>
      <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="email" />
      <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="password" />
      <button onClick={handleRegister}>Register</button>
      {error && <p>{error}</p>}
    </div>
  )
}
