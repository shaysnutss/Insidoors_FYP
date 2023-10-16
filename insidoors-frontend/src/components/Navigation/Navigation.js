import "./Navigation.css"
import { Logo, Logout } from "..";
import React from "react";
import { useNavigate } from "react-router-dom";

const Navigation = () => {
    const navigate = useNavigate();
    return(
        <div className="nav">
            <div className = "screen">
                <div className="logo">
                    <Logo></Logo>
                </div>
                <div className="navigation-tab">
                    <div>
                        <button className="dashboard-tab" onClick={() =>
                        navigate("/main/dashboard")}>Dashboard</button>
                    </div>
                    <div>
                        <button className="employee-tab">Employees</button>
                    </div>
                    <div>
                        <button className="case-tab" onClick={() =>
                        navigate("/main/case")}>Case Management</button>
                    </div>
                </div>
                <div className="extra-tab">
                    <div>
                        {/* <button className="alert-tab">Alerts</button> */}
                    </div>
                    <div className="logout">
                        <Logout></Logout>
                    </div>
                </div>
            </div>

        </div>
        
        
    )
}

export default Navigation