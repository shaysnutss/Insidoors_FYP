import "./Navigation.css"

const Navigation = () => {
    return(
        <div className="navigation">
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
            </div>

        </div>
    )
}

export default Navigation