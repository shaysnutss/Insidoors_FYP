import "./Dashboard.css";
import { Logo, Logout } from ".."
import { useState, useEffect } from "react";
import userService from "../../services/user.service";
import authService from "../../services/auth.service";
import { useNavigate } from "react-router-dom";
import Modal from "../Case/Modal/Modal";
import { line } from "../../assets"

import {
  TableauViz
} from 'https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js';

const Dashboard = () => {
  const [name, setName] = useState("");
  const [cases, setCases] = useState([]);
  const [currentUser, setCurrentUser] = useState([]);
  const [openCases, setOpenCases] = useState(0);
  const [highCaseId, setHighCaseId] = useState(0);
  const [assignCases, setAssignCases] = useState(0);
  const [detail, setDetail] = useState([]);
  const navigate = useNavigate();
  const [modalOpen, setModalOpen] = useState(false);
  const viz = new TableauViz();
  let sumOpen = 0;
  let sumAssign = 0;
  let severity = 0;
  let caseId = 0;

  useEffect(() => {
    viz.src = 'https://public.tableau.com/views/Dashboard-Insidoors/Storyboard?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link';
    viz.toolbar = 'hidden';
    viz.device = 'desktop';
    viz.width = '1366px';
    viz.height = '768px';

    document.getElementById('tableauViz').appendChild(viz);

    try {
      userService.getAccountById().then(
        () => {
          fetchCases();
          getCurrentUser();
          fetchName();
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
  }, []);

  function getCurrentUser() {
    userService.getUser()
      .then(function (response) {
        setCurrentUser(response.data);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      })
  }

  useEffect(() => {
    cases.forEach(countOpenCases);
    setOpenCases(sumOpen);
    setAssignCases(sumAssign);
    setHighCaseId(caseId);
  }, [cases])

  useEffect(() => {
    userService.getCaseById(highCaseId)
      .then(function (response) {
        setDetail(response.data);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      })
  }, [highCaseId])

  function countOpenCases(item) {
    if (item.status === "Open") {
      sumOpen += 1;

      if (item.severity > severity) {
        severity = item.severity;
        caseId = item.id;
      }
    }

    if ((item.status === "Assigned") && (item.socName === currentUser.name)) {
      sumAssign += 1;
    }
  }

  const fetchName = async () => {
    const { data } = await userService.getName();
    const name1 = data;
    const nameCapitalized = name1.charAt(0).toUpperCase() + name1.slice(1);
    setName(nameCapitalized);
  }

  function fetchCases() {
    userService.getAllCases()
      .then(function (response) {
        setCases(response.data);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      })
  };

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
        <div className="name">
          Hello, {name}!
        </div>
        <div className="open-ticket">
          <div className="ticket-title">Open Cases</div>
          <div className="ticket-size">
            <div className="ticket">
            <div className="open-text">{openCases}</div>
            </div>
          </div>
        </div>
        <div className="assign-ticket">
          <div className="ticket-title">Assigned To Me</div>
          <div className="ticket-size">
            <div className="ticket">
              <div className="open-text">{assignCases}</div>
            </div>
          </div>
        </div>
        <div className="priority-ticket">
          <div className="ticket-title">Urgent</div>
          <div>
            <div className="ticket-size">
              <div className="ticket">
                <img className="line" alt="" src={line} />
                <div className="ticket-body">
                  {detail.severity <= 50 &&
                    <div className="priority-low">Low</div>
                  }
                  {detail.severity > 50 && detail.severity <= 100 &&
                    <div className="priority-med">Med</div>
                  }
                  {detail.severity > 100 &&
                    <div className="priority-high">High</div>
                  }
                  <div className="title">{detail.incidentTitle}</div>
                  <div className="description">{detail.incidentDesc}</div>
                </div>
                <div className="person" >{detail.socName}</div>
                <button className="message" onClick={() => { setModalOpen(true); }}>View {">"}</button>
              </div>
            </div>
          </div>
        </div>
        <div className="visual" id="tableauViz" />
        {modalOpen && <Modal setOpenModal={setModalOpen} caseId={highCaseId} />}
      </div>
    </div>
  );
};
export default Dashboard