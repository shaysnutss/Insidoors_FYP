import "./Navigation.css"
import { Logo, Logout } from "..";
import React from "react";
import { useNavigate } from "react-router-dom";

const Navigation = ({resetEmployees}) => {
    const navigate = useNavigate();
    const handleViewAllEmployees = () => {
        resetEmployees(); // Call the function passed as a prop to reset employees
        navigate("/main/employees");
    };

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
                        <button className="employee-tab" onClick={handleViewAllEmployees} >Employees</button>
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