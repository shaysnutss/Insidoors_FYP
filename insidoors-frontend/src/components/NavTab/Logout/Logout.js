import "./Logout.css";
import authService from "../../../services/auth.service";
import { useNavigate } from "react-router-dom";
import React from 'react'

const Logout = () => {
    const navigate = useNavigate();
    
    return (
        <button className="logout-tab" onClick={() => {
            authService.logout();
            navigate("/auth/login");
        }
        }>Logout</button>
    );
};

export default Logout