import "./Dashboard.css";
import { Logo} from ".."
import { useEffect } from "react";
import userService from "../../services/user.service";
import authService from "../../services/auth.service";
import { useNavigate } from "react-router-dom";
import Tableau from "tableau-react";

const Dashboard = () => {
  const navigate = useNavigate();

  const options = {
    height: 768,
    width: 1366,
  };

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
            <button className="employee-tab">Employees</button>
          </div>
          <div>
            <button className="case-tab" onClick={() =>
              navigate("/main/case")}>Case Management</button>
          </div>
        </div>
        <div className="extra-tab">
          <div>
            <button className="alert-tab">Alerts</button>
          </div>
          <div>
          </div>
        </div>
        <div className="visual">
          <Tableau
            url="https://public.tableau.com/views/Dashboard-PCAccessLogs/PCAccessLogs?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link"
            options={options}
          />
        </div>
      </div>
    </div>

  );
};
export default Dashboard