import "./Dashboard.css";
import { useEffect } from "react";
import userService from "../../services/user.service";
import authService from "../../services/auth.service";
import { useNavigate } from "react-router-dom";
import Tableau from "tableau-react";

const Dashboard = () => {
  const navigate = useNavigate();

  useEffect(() => {
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
  });

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
        <div className="visual">
          <Tableau
            url="https://public.tableau.com/views/Dashboard-PCAccessLogs/PCAccessLogs?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link"
          />
        </div>
      </div>
    </div>

  );
};
export default Dashboard