import "./Dashboard.css";
import React, { useState, useEffect } from "react";
import userService from "../../services/user.service";
import authService from "../../services/auth.service";
import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const [privatePosts, setPrivatePosts] = useState([]);
  let message = "boby";
  const navigate = useNavigate();

  useEffect(() => {
    userService.getAccountById().then(
      (response) => {
        setPrivatePosts(response.data);
        console.log("successful");
      },
      (error) => {
        console.log("Private page", error.response);
        // Invalid token
        if (error.response && error.response.status === 403) {
          authService.logout();
          navigate("/auth/login");
          window.location.reload();
        }
      }
    );
  }, []);

    return (
        <div className="dashboard">
            <div className="div">
                <div className="overlap">
                    <div className="ellipse" />
                    <div className="text-wrapper">Insidoors</div>
                </div>
                <div className="overlap-group">
                    <div className="text-wrapper-2">Dashboard</div>
                    <div className="text-wrapper-3">Employees</div>
                    <div className="text-wrapper-4">Case Management</div>
                </div>
                <div className="overlap-2">
                    <div className="text-wrapper-5">Logout</div>
                    <div className="text-wrapper-6">Alerts</div>
                </div>
            </div>
        </div>
    );
};
export default Dashboard