import React from "react";
import "./Register.css";
import { Link } from "react-router-dom";
import { registerPic } from "../../assets"

export const Register = () => {
  return (
    <div className="register">
      <div className="div">
        <div className="overlap">
          <div className="ellipse" />
          <div className="text-wrapper">Insidoors</div>
        </div>
        <div className="overlap-group">
          <div className="overlap-group-2">
            <div className="text-wrapper-2">Create new account.</div>
            <p className="already-have-an">
              <span className="span">Already have an account?</span>
              <span className="text-wrapper-3">&nbsp;</span>
              <span><Link className="text-wrapper-4" to="/login" style={{textDecoration:'none'}}>Log in</Link></span>
            </p>
            <img className="login-rafiki" alt="Login rafiki" src={registerPic} />
          </div>
          <div className="input-button">
            <button className="button">Create Account</button>
          </div>
          <div>
            <input type="text" className="text-input" placeholder="Name" name="nameReg" />
          </div>
          <div>
            <input type="text" className="text-input" placeholder="E-mail" name="emailReg" />
          </div>
          <div>
            <input type="password" className="text-input" placeholder="Password" name="passwordReg" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Register