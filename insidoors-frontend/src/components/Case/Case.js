import React, { useState, useEffect } from "react";
import userService from "../../services/user.service";
import "./Case.css"
import { openCases, assigned, inReview, closed, plus, message, line } from "../../assets"

const Case = () => {
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
      <div className="div">
        <div className="overlap">
          <div className="ellipse" />
          <div className="text-wrapper">Insidoors</div>
        </div>
        <div className="overlap-group">
          <div className="text-wrapper-2">Logout</div>
          <div className="text-wrapper-3">Alerts</div>
        </div>
        <div className="overlap-2">
          <div className="text-wrapper-4">Employees</div>
          <div className="text-wrapper-5">Case Management</div>
          <div className="text-wrapper-3">Dashboard</div>
        </div>
        <div className="overlap-3">
          <div className="frame">
            <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {
                    <div className="ticket-high-priority">
                      <img className="line" alt="Line" src={line} />
                      <div className="overlap-4">
                        <div className="overlap-group-2">
                          <div className="rectangle" />
                          <div className="text-wrapper-6">High</div>
                        </div>
                        <div className="overlap-group-3">
                          <div className="text-wrapper-8">{cases.incidentTitle}</div>
                          <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                        </div>
                      </div>
                      <img className="image" alt="" src={message} />
                      <img className="img" alt="" src={plus} />
                    </div>
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text-wrapper-9">Open Cases</div>
          <img className="image-2" alt="" src={openCases} />
        </div>
        <div className="overlap-5">
          <div className="ticket-high-priority-wrapper">
          <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {cases.id === 3 &&
                    <div className="ticket-high-priority">
                      <img className="line" alt="Line" src={line} />
                      <div className="overlap-4">
                        <div className="overlap-group-2">
                          <div className="rectangle" />
                          <div className="text-wrapper-6">High</div>
                        </div>
                        <div className="overlap-group-3">
                          <div className="text-wrapper-8">{cases.incidentTitle}</div>
                          <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                        </div>
                      </div>
                      <img className="image" alt="" src={message} />
                      <img className="img" alt="" src={plus} />
                    </div>
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text-wrapper-9">Assigned</div>
          <img className="image-3" alt="" src={assigned} />
        </div>
        <div className="overlap-6">
          <div className="div-wrapper">
          <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {cases.status == "Closed" &&
                    <div className="ticket-high-priority">
                      <img className="line" alt="Line" src={line} />
                      <div className="overlap-4">
                        <div className="overlap-group-2">
                          <div className="rectangle" />
                          <div className="text-wrapper-6">High</div>
                        </div>
                        <div className="overlap-group-3">
                          <div className="text-wrapper-8">{cases.incidentTitle}</div>
                          <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                        </div>
                      </div>
                      <img className="image" alt="" src={message} />
                      <img className="img" alt="" src={plus} />
                    </div>
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text-wrapper-9">In Review</div>
          <img className="image-4" alt="" src={inReview} />
        </div>
        <div className="overlap-7">
          <div className="frame-2">
          <div>
              {cases.map((cases) => (
                <p key={cases.id}>
                  {cases.status == "Open" &&
                    <div className="ticket-high-priority">
                      <img className="line" alt="Line" src={line} />
                      <div className="overlap-4">
                        <div className="overlap-group-2">
                          <div className="rectangle" />
                          <div className="text-wrapper-6">High</div>
                        </div>
                        <div className="overlap-group-3">
                          <div className="text-wrapper-8">{cases.incidentTitle}</div>
                          <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                        </div>
                      </div>
                      <img className="image" alt="" src={message} />
                      <img className="img" alt="" src={plus} />
                    </div>
                  }
                </p>
              ))}
            </div>
          </div>
          <div className="text-wrapper-9">Closed</div>
          <img className="image-5" alt="" src={closed} />
        </div>
        <div className="overlap-8">
          <div>
            <input type="text" className="rectangle-2" placeholder="Search" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Case
