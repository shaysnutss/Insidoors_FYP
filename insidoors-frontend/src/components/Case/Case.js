import React, { useState, useEffect } from "react";
import { Logo, Logout } from "..";
import { useNavigate } from "react-router-dom";
import authService from "../../services/auth.service";
import userService from "../../services/user.service";
import Modal from "./Modal/Modal";
import "./Case.css";
import { openCases, assigned, inReview, closed, plus, line, male, female } from "../../assets"

const Case = () => {
  const navigate = useNavigate();
  const [cases, setCases] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [caseId, setCaseId] = useState(0);

  function Ticket(cases) {
    return (<div className="ticket">
      <img className="line" alt="" src={line} />
      <div className="ticket-body">
        {cases.severity === 50 &&
          <div className="priority-low">Low</div>
        }
        {cases.severity === 100 &&
          <div className="priority-med">Med</div>
        }
        {cases.severity === 200 &&
          <div className="priority-high">High</div>
        }
        <div className="title">{cases.title}</div>
        <p className="description">{cases.employeeFirstname} {cases.employeeLastname}{"\n"}{cases.incidentTimestamp}</p>
      </div>
      <img className="person" alt="" src={plus} />
      <button className="message" onClick={() => { setModalOpen(true); setCaseId(cases.id); }}>View {">"}</button>
    </div>)
  }

  const fetchCases = async (e) => {
    const { data } = await userService.getAllCases();
    const cases = data;
    setCases(cases);
    console.log(cases);
  };

  useEffect(() => {
    fetchCases();
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
  }, []);

  return (
    <div className="case">
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
            {/* <button className="alert-tab">Alerts</button> */}
          </div>
          <div className="logout">
            <Logout></Logout>
          </div>
        </div>
        <div className="open-cases">
          <div className="open-cases-container">
            <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {/* cases.status === "Open" && */
                    Ticket(cases)
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text">Open Cases</div>
          <img className="pic" alt="" src={openCases} />
        </div>
        <div className="assigned-cases">
          <div className="assigned-cases-container">
            <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {cases.status === "Assigned" &&
                    Ticket(cases)
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text">Assigned</div>
          <img className="pic" alt="" src={assigned} />
        </div>
        <div className="review-cases">
          <div className="review-cases-container">
            <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {cases.status === "In review" &&
                    Ticket(cases)
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text">In Review</div>
          <img className="pic" alt="" src={inReview} />
        </div>
        <div className="closed-cases">
          <div className="closed-cases-container">
            <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {cases.status === "Closed" &&
                    Ticket(cases)
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text">Closed</div>
          <img className="pic" alt="" src={closed} />
        </div>
        <div>
          <input type="text" className="searchbar" placeholder="Search" />
        </div>
      </div>
      {modalOpen && <Modal setOpenModal={setModalOpen} cases={caseId} />}
    </div >
  );
};

export default Case
