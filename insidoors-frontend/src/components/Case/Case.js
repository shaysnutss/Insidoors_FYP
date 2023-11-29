import React, { useState, useEffect } from "react";
import { Logo, Logout } from "..";
import { useNavigate } from "react-router-dom";
import authService from "../../services/auth.service";
import userService from "../../services/user.service";
import Modal from "./Modal/Modal";
import "./Case.css";
import { openCases, assigned, inReview, closed, line } from "../../assets"

const Case = () => {
  const navigate = useNavigate();
  const [cases, setCases] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [caseId, setCaseId] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [socList, setSocList] = useState([]);
  const [filteredData, setFilteredData] = useState(cases);
  const [currentUser, setCurrentUser] = useState([]);
  const [filteredDataSecond, setFilteredDataSecond] = useState(cases);


  function Ticket(cases) {
    return (
      <div className="ticket-size">
        <div className="ticket">
          <img className="line" alt="" src={line} />
          <div className="ticket-body">
            {cases.severity <= 50 &&
              <div className="priority-low">Low</div>
            }
            {cases.severity > 50 && cases.severity <= 100 &&
              <div className="priority-med">Med</div>
            }
            {cases.severity > 100 &&
              <div className="priority-high">High</div>
            }
            <div className="title">{cases.incidentTitle}</div>
            <div className="description">{cases.incidentDesc}</div>
          </div>
          <div className="person" >{cases.socName}</div>
          <button className="message" onClick={() => { setModalOpen(true); setCaseId(cases.id); }}>View {">"}</button>
        </div>
      </div>
    )
  }

  function getSoc() {
    userService.getAllSoc()
      .then(function (response) {
        setSocList(response.data);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      })
  }

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

  function fetchCases() {
    userService.getAllCases()
      .then(function (response) {
        setCases(response.data);
        // KGB
        setFilteredData(response.data);
        setFilteredDataSecond(response.data);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      })
  };

  function setCurrentCases() {
    const filter = cases.filter((item) =>
      item.accountId === currentUser.id || item.status === "Open"
    )
    setFilteredData(filter);
    setFilteredDataSecond(filter);
  }

  useEffect(() => {
    setCurrentCases();
  },[cases])

  useEffect(() => {
    try {
      userService.getAccountById().then(
        () => {
          fetchCases();
          getSoc();
          getCurrentUser();
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

  const handleSearch = (e) => {
    const { value } = e.target;
    setSearchTerm(value);
    console.log(value);
    filterData(value);
  }

  const filterData = (searchTerm) => {
    const filter = filteredDataSecond.filter((item) =>
      item.incidentTitle.toLowerCase().includes(searchTerm.toLowerCase()) || item.incidentDesc.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredData(filter);
  };

  function handleSoc(e) {
    if (e.target.value === "reset") {
      setFilteredDataSecond(cases);
      setFilteredData(cases);
    }
    else {
      const filter = cases.filter((item) =>
        item.accountId == (e.target.value) || item.status === "Open"
      )
      setFilteredData(filter);
      setFilteredDataSecond(filter);
    }
  };

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
          <div className="logout">
            <Logout></Logout>
          </div>
        </div>
        <div className="open-cases">
          <div className="open-cases-container">
            <div>
              {filteredData.map((cases) => (
                <div key={cases.id}>
                  {cases.status === "Open" &&
                    Ticket(cases)
                  }
                </div>
              ))}
            </div>
          </div>
          <div className="text">Open Cases</div>
          <img className="pic" alt="" src={openCases} />
        </div>
        <div className="assigned-cases">
          <div className="assigned-cases-container">
            <div>
              {filteredData.map((cases) => (
                <div key={cases.id}>
                  {cases.status === "Assigned" &&
                    Ticket(cases)
                  }
                </div>
              ))}
            </div>
          </div>
          <div className="text">Assigned</div>
          <img className="pic" alt="" src={assigned} />
        </div>
        <div className="review-cases">
          <div className="review-cases-container">
            <div>
              {filteredData.map((cases) => (
                <div key={cases.id}>
                  {cases.status === "In review" &&
                    Ticket(cases)
                  }
                </div>
              ))}
            </div>
          </div>
          <div className="text">In Review</div>
          <img className="pic" alt="" src={inReview} />
        </div>
        <div className="closed-cases">
          <div className="closed-cases-container">
            <div>
              {filteredData.map((cases) => (
                <div key={cases.id}>
                  {cases.status === "Closed" &&
                    Ticket(cases)
                  }
                </div>
              ))}
            </div>
          </div>
          <div className="text">Closed</div>
          <img className="pic" alt="" src={closed} />
        </div>
        <div>
          <input type="text" className="searchbar" placeholder="Search" value={searchTerm} onChange={handleSearch} />
        </div>
        <select className="socSelect" onChange={handleSoc}>
          <option value="reset">Everyone</option>
          {socList.map((socList) => (
            <option key={socList.id} value={socList.id} selected={currentUser.id === socList.id}>{socList.socName}</option>
          ))}
        </select>
      </div>
      {modalOpen && <Modal setOpenModal={setModalOpen} caseId={caseId} />}
    </div >
  );
};

export default Case
