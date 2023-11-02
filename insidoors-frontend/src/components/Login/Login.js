import { loginPic } from "../../assets"
import { Logo } from "..";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import authService from "../../services/auth.service";
import "./Login.css";
import React from 'react'

const Login = () => {

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
        <div className="screen">
          <div className="logo">
            <Logo></Logo>
          </div>
          <div className="page">
            <div className="title">
              <div className="title-text">Log In.</div>
              <img className="image" alt="" src={loginPic} />
            </div>
            <div>
              <input type="text" className="input-email" placeholder="E-mail" value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <input type="password" className="input-password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
            </div>
            <div>
              <button type="submit" className="submit-button">Log in</button>
            </div>
          </div>
        </div>
      </div>
    </form>

  );
};
export default Login