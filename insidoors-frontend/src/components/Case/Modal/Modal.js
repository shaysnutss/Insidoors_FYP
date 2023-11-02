import "./Modal.css";
import React, { useState, useEffect } from "react";
import userService from "../../../services/user.service"
import { line } from "../../../assets"

function Modal({ setOpenModal, caseId }) {

    const [comments, setComments] = useState([]);
    const [detail, setDetail] = useState([]);
    const [tab, setTab] = useState("modal1");
    const [openSoc,setOpenSoc] = useState(false);

    const fetchComments = async (e) => {
        const { data } = await userService.getAllCases();
        const comments = data;
        setComments(comments);
        console.log(comments);
    };

    const fetchCase = async (e) => {
        const { data } = await userService.getSingleCase(caseId);
        const detail = data;
        setDetail(detail);
        console.log(detail);
      };

    const handleOpen = () => {
        setOpenSoc()
    }

    useEffect(() => {
        fetchComments();
        fetchCase();
    }, []);

    return (
        <div className="background">
            <div>
                <div className={tab}>
                    <button className="closeButton" onClick={() => { setOpenModal(false); }}> x </button>
                    <img className="lineTop" alt="" src={line} />
                    <div className="textTitle">Title</div>
                    <div className="textEmployee">Employee</div>
                    <div className="textIncident">Incident Timestamp</div>
                    <div className="textSeverity">Severity</div>
                    <div className="textStatus">Status</div>
                    <div className="textSoc">SOC</div>
                    <div className="textDate">Date Assigned</div>
                    <div className="employee">Brooklyn Simmons</div>
                    <div className="incident">06/08/23 00:00:00</div>
                    <div className="severity">High</div>
                    <div className="status">In Review</div>
                    <div className="soc">Zack</div>
                    <div className="date">August 27, 2023</div>
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
                                            <p key={comments.id}>
                                                {/* cases.status === "Open" && */
                                                    <div className="commentBox">
                                                        <div className="commentTitle">Zack</div>
                                                        <div className="commentDescription">
                                                            This ticket needs to be solved asap please as it involves major security issues. Thank you very much.
                                                        </div>
                                                        <div className="rectangle" />
                                                    </div>
                                                }
                                            </p>
                                        ))}
                                    </div>
                                </div>

                                <div className="addCommentBox">
                                    <div className="addCommentTitle">Enter name</div>
                                    <div className="addCommentDescription">Enter description</div>
                                </div>
                            </div>
                        }
                    </div>
                    <div>
                        {tab === "modal1" &&
                            <div className="descBox">This will be the description of the ticket </div>
                        }
                    </div>
                    <button className="saveButton">
                        <div className="saveButtonText">Save</div>
                    </button>
                </div>
            </div>
        </div>
    );
}

export default Modal;