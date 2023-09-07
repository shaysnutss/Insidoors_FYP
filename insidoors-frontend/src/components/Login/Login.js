import { loginPic } from "../../assets"
import "./Login.css";

export const Login = () => {
  return (
    <div className="login">
      <div className="div">
        <div className="overlap-group">
          <div className="ellipse" />
          <div className="text-wrapper">Insidoors</div>
        </div>
        <div className="overlap">
          <div className="overlap-2">
            <div className="text-wrapper-2">Log in.</div>
            <p className="don-t-have-an">
              <span className="span">Don’t have an account?</span>
              <span><a href="register"> Register</a></span>
            </p>
            <img className="login-cuate" alt="Login cuate" src={loginPic} />
          </div>
          <div>
            <button className="input-button">Log in</button>
          </div>
          <div>
            <input type="text" className="input-text" placeholder="E-mail" name="email" />
          </div>
          <div>
            <input type="password" className="text-input-wrapper" placeholder="Password" name="password" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login