import "./Modal.css";
import React, { useState, useEffect } from "react";
import userService from "../../../services/user.service"
import { useNavigate } from "react-router-dom";
import { line } from "../../../assets"

function Modal({ setOpenModal, caseId }) {

    const navigate = useNavigate();
    const [comments, setComments] = useState([]);
    const [detail, setDetail] = useState([]);
    const [socList, setSocList] = useState([]);
    const [tab, setTab] = useState("modal1");
    const [status, setStatus] = useState("");
    const [soc, setSoc] = useState("");
    const [socId, setSocId] = useState(0);
    const [priority, setPriority] = useState("");
    const [desc, setDesc] = useState("");
    const [checked, setChecked] = useState(false);

    function fetchComments() {
        userService.getAllCommentsById(caseId)
            .then(function (response) {
                setComments(response.data);
            })
            .catch(function (error) {
                // handle error
                console.log(error);
            })
    };

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

    function getAllCases() {
        userService.getCaseById(caseId)
            .then(function (response) {
                setDetail(response.data);
            })
            .catch(function (error) {
                // handle error
                console.log(error);
            })
    }

    function savingModal() {
        if (desc !== "") {
            console.log("comment");
        }

        if (checked === true) {
            userService.closeCase(caseId, detail.severity)
                .catch(function (error) {
                    // handle error
                    console.log(error);
                })
        }

        userService.putSoc(caseId, socId)
            .catch(function (error) {
                // handle error
                console.log(error);
            })

        userService.putStatus(caseId, status)
            .catch(function (error) {
                // handle error
                console.log(error);
            })

        navigate("/main/case");
        window.location.reload();
    }

    function handleSoc(e) {
        const selectedIndex = e.target.options.selectedIndex;
        setSoc(e.target.value);
        setSocId(e.target.options[selectedIndex].getAttribute('data-key'));
    };

    function handleStatus(e) {
        setStatus(e.target.value);
    };

    function handleChecked() {
        setChecked(!checked);
    }

    useEffect(() => {
        getAllCases();
        getSoc();
        fetchComments();
    }, [])

    useEffect(() => {
        if (detail.severity <= 50) {
            setPriority("Low");
        } else if (detail.severity > 50 && detail.severity <= 100) {
            setPriority("Med");
        } else if (detail.severity > 100) {
            setPriority("High");
        }
        setSoc(detail.socName);
        setSocId(detail.accountId);
        setStatus(detail.status);
    }, [detail])


    return (
        <div className="background">
            <div>
                <div className={tab}>
                    <button className="closeButton" onClick={() => { setOpenModal(false); }}> x </button>
                    <img className="lineTop" alt="" src={line} />
                    <div className="textTitle">{detail.incidentTitle}</div>
                    <div className="textEmployee">Employee</div>
                    <div className="textIncident">Incident Timestamp</div>
                    <div className="textSeverity">Severity</div>
                    <div className="textStatus">Status</div>
                    <div className="textSoc">SOC</div>
                    <div className="textDate">Date Assigned</div>
                    <div className="employee">{detail.employeeFirstname} {detail.employeeLastname}</div>
                    <div className="incident">{detail.incidentTimestamp}</div>
                    <div className="severity">{priority}</div>
                    <select className="status" value={status} onChange={handleStatus}>
                        <option value="Open">Open</option>
                        <option value="Assigned">Assigned</option>
                        <option value="In review">In review</option>
                        <option value="Closed">Closed</option>
                    </select>
                    <select className="soc" value={soc} data-key={socId} onChange={handleSoc}>
                        {socList.map((socList) => (
                            <option key={socList.id} data-key={socList.id} value={socList.socName}>{socList.socName}</option>
                        ))}
                    </select>
                    <div className="date">{detail.dateAssigned}</div>
                    <div className="tab">
                        <button className="descSelected" onClick={() => { setTab("modal1"); }}>DESCRIPTION</button>
                        <button className="commentSelected" onClick={() => { setTab("modal"); }}>COMMENTS</button>
                        <img className="lineBot" alt="" src={line} />
                    </div>
                    <div>
                        {tab === "modal" &&
                            <div className="comments">
                                <div className="commentsGet">
                                    <div>
                                        {comments.map((comments) => (
                                            <div className="box-size" key={comments.id}>
                                                {
                                                    <div className="commentBox">
                                                        <div className="commentTitle">{comments.socName}</div>
                                                        <div className="commentDescription">
                                                            {comments.commentDescription}
                                                        </div>
                                                        <div className="rectangle" />
                                                    </div>
                                                }
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="addCommentBox">
                                    <div className="addCommentTitle"></div>
                                    <input type="text" className="addCommentDescription" placeholder="Enter Description" value={desc} onChange={(e) => setDesc(e.target.value)} />
                                </div>
                            </div>
                        }
                    </div>
                    <div>
                        {tab === "modal1" &&
                            <div className="descBox">{detail.incidentDesc}</div>
                        }
                    </div>
                    {status === "Closed" &&
                        <label className="checkbox"><input type="checkbox" checked={checked} onChange={handleChecked}/>True Positive</label>
                    }
                    <button className="saveButton" onClick={savingModal}> Save </button>
                </div>
            </div>
        </div>
    );
}

export default Modal;
