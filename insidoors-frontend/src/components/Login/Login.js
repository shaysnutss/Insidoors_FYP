import { loginPic } from "../../assets"
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import authService from "../../services/auth.service";
import "./Login.css";

export const Login = () => {

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await authService.login(email, password).then(
        () => {
          navigate("/main/dashboard");
          window.location.reload();
        },
        (error) => {
          console.log(error);
        }
      );
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <form onSubmit={handleLogin}>
      <div className="login">
        <div className="div">
          <div className="overlap-group">
            <div className="ellipse" />
            <div className="text-wrapper">Insidoors</div>
          </div>
          <div className="overlap">
            <div className="overlap-2">
              <div className="text-wrapper-2">Log in.</div>
              <img className="login-cuate" alt="Login cuate" src={loginPic} />
            </div>

            <div>
              <button type="submit" className="input-button">Log in</button>
            </div>
            <div>
              <input type="text" className="input-text" placeholder="E-mail" value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <input type="password" className="text-input-wrapper" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
            </div>
          </div>
        </div>
      </div>
    </form>

  );
};
export default Login