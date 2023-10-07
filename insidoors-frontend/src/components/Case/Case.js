import React, { useState, useEffect } from "react";
import { Logo, Logout } from "..";
import { useNavigate } from "react-router-dom";
import userService from "../../services/user.service";
import "./Case.css"
import { openCases, assigned, inReview, closed, plus, message, line } from "../../assets"

const Case = () => {
  const navigate = useNavigate();
  const [cases, setCases] = useState([]);

  const fetchCases = async (e) => {
    const { data } = await userService.getAllCases();
    const cases = data;
    setCases(cases);
    console.log(cases);
  };

  useEffect(() => {
    fetchCases();
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
          <div className="logout">
            <Logout></Logout>
          </div>
        </div>
        <div className="open-cases">
          <div className="open-cases-container">
            <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {
                    <div className="ticket">
                      <img className="line" alt="" src={line} />
                      <div className="ticket-body">
                        <div className="priority">High</div>
                        <div className="title">{cases.title}</div>
                        <p className="description">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                      </div>
                      <img className="person" alt="" src={message} />
                      <img className="message" alt="" src={plus} />
                    </div>
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
                  {
                    <div className="ticket">
                      <img className="line" alt="" src={line} />
                      <div className="ticket-body">
                        <div className="priority">High</div>
                        <div className="title">{cases.title}</div>
                        <p className="description">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                      </div>
                      <img className="person" alt="" src={message} />
                      <img className="message" alt="" src={plus} />
                    </div>
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
                  {
                    <div className="ticket">
                      <img className="line" alt="" src={line} />
                      <div className="ticket-body">
                        <div className="priority">High</div>
                        <div className="title">{cases.title}</div>
                        <p className="description">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                      </div>
                      <img className="person" alt="" src={message} />
                      <img className="message" alt="" src={plus} />
                    </div>
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
                  {
                    <div className="ticket">
                      <img className="line" alt="" src={line} />
                      <div className="ticket-body">
                        <div className="priority">High</div>
                        <div className="title">{cases.title}</div>
                        <p className="description">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                      </div>
                      <img className="person" alt="" src={message} />
                      <img className="message" alt="" src={plus} />
                    </div>
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
    </div>
  );
};

export default Case
