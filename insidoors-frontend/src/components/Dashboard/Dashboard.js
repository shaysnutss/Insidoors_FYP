import "./Dashboard.css";
import { Logo, Logout } from ".."
import React, { useState, useEffect } from "react";
import userService from "../../services/user.service";
import authService from "../../services/auth.service";
import { useNavigate } from "react-router-dom";
import {
  TableauViz,
  TableauEventType
} from 'https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js';

const Dashboard = () => {
  const [name,setName] = useState([]);
  const navigate = useNavigate();
  const viz = new TableauViz();

  const options = {
    height: 768,
    width: 1366,
  };

  useEffect(() => {
    viz.src = 'https://public.tableau.com/views/Dashboard-PCAccessLogs/PCAccessLogs?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link';
    viz.toolbar = 'hidden';
    viz.device = 'desktop';
    viz.width = '1366px';
    viz.height = '768px';

    document.getElementById('tableauViz').appendChild(viz);

    try {
      userService.getAccountById().then(
        () => {
          console.log("ok");
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
    } catch (err) {
      console.log(err);
      navigate("/auth/login");
    }
    
    fetchName();


  }, []);

  const fetchName = async () => {
    const {data} = await userService.getName();
    const name = data;
    const nameCapitalized = name.charAt(0).toUpperCase() + name.slice(1);
    setName(nameCapitalized);
    console.log(nameCapitalized);
  }

  const changeViewBuilding = () => {
    viz.src = 'https://public.tableau.com/views/Dashboard-BuildingAccessLogs/BuildingAccessLogs?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link';
    document.getElementById('tableauViz').appendChild(viz);
  }

  const changeViewPC = () => {
    viz.src = 'https://public.tableau.com/views/Dashboard-PCAccessLogs/PCAccessLogs?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link';
    document.getElementById('tableauViz').appendChild(viz);
  }

  return (
    <div className="dashboard">
      <div className="screen">
        <div className="logo">
          <Logo></Logo>
        </div>
        <div className="navigation-tab">
          <div>
            <button className="dashboard-tab" onClick={() =>
              navigate("/main/dashboard")}>Dashboard</button>
          </div>
          <div>
            <button className="employee-tab" onClick={() =>
              navigate("/main/employees")}>Employees</button>
          </div>
          <div>
            <button className="case-tab" onClick={() =>
              navigate("/main/case")}>Case Management</button>
          </div>
        </div>
        <div className="extra-tab">
          <div>
            <Logout></Logout>
          </div>
        </div>
        <div className = "name">
            Hello, {name}!
        </div>
        <div className = "dashboardView">
          <button className="change-to-pc" onClick={changeViewPC}>PC</button>
          <button className="change-to-building" onClick={changeViewBuilding}>Building</button>
          <div className="visual" id="tableauViz" />
        </div>
        
      </div>
    </div>

  );
};
export default Dashboard